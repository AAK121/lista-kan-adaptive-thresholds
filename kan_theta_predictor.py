"""
kan_theta_predictor.py
KAN-based adaptive threshold predictor for LISTA-CP-KAN.

Two variants:
  '1norm' : input = log1p(L2/sqrt(M))     -> shape (batch, 1)
  '3norm' : input = [log1p(L1/M), log1p(L2/sqrt(M)), log1p(Linf)] -> shape (batch, 3)

Log1p transform (Fix 3): compresses wide dynamic range of norms across
layers into a compact interval that the spline grid can cover properly.
Norms shrink by orders of magnitude from layer 1 to layer 16; without
log-scaling, late-layer inputs fall far outside the spline grid range.

Output pipeline:
  residual -> compute_norms -> log1p -> KAN -> clamp[-10,5] -> +bias -> softplus -> theta
Clamp (Fix 4) prevents runaway thresholds during unstable early epochs.
Softplus enforces theta > 0 for soft thresholding.

initialise_to_constant(c):
  Zeros spline coefs, measures actual KAN residual output, and sets
  theta_bias to exactly compensate so forward() returns c for any input.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN


class KANThetaPredictor(nn.Module):
    """
    Adaptive threshold predictor using a single-layer KAN.

    Parameters
    ----------
    variant : '1norm' or '3norm'
    M       : measurement dimension (used for normalisation; default 250)
    device  : torch device string
    """

    def __init__(self, variant: str = '1norm', M: int = 250, device: str = 'cpu',
                 width: list = None):
        super().__init__()
        assert variant in ('1norm', '3norm'), \
            "variant must be '1norm' or '3norm'"
        self.variant = variant
        self.M = M
        if width is None:
            width = [1, 2, 1] if variant == '1norm' else [3, 6, 1]

        self.kan = KAN(
            width=width,
            grid=5,
            k=3,
            grid_range=[-2, 4],
            device=device,
            auto_save=False,
            symbolic_enabled=False,
            save_act=False,
        )

        # Additive bias after the KAN. Set by initialise_to_constant so
        # that forward() returns the baseline theta value at warm-start.
        self.theta_bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Norm computation  (Fix 3: log1p transform applied here)
    # ------------------------------------------------------------------
    def compute_norms(self, r: torch.Tensor) -> torch.Tensor:
        """
        r : (batch, M) residual tensor.
        Computes normalised norms, then applies log1p to compress the
        dynamic range across layers (norms vary by orders of magnitude).

        Returns:
          '1norm' -> (batch, 1)  : log1p(L2/sqrt(M))
          '3norm' -> (batch, 3)  : [log1p(L1/M), log1p(L2/sqrt(M)), log1p(Linf)]
        """
        m = float(self.M)
        if self.variant == '1norm':
            l2 = torch.norm(r, p=2, dim=1, keepdim=True) / math.sqrt(m)
            return torch.log1p(l2)                       # (batch, 1)
        else:
            l1   = torch.norm(r, p=1,   dim=1, keepdim=True) / m
            l2   = torch.norm(r, p=2,   dim=1, keepdim=True) / math.sqrt(m)
            linf = torch.norm(r, p=float('inf'), dim=1, keepdim=True)
            raw  = torch.cat([l1, l2, linf], dim=1)      # (batch, 3)
            return torch.log1p(raw)

    # ------------------------------------------------------------------
    # Initialisation to constant output  (Fix 2: calibrated warm-start)
    # ------------------------------------------------------------------
    def initialise_to_constant(self, c: float):
        """
        Configure so that forward() returns exactly c for any input.

        Strategy:
          1. Zero spline coefs -> spline contribution = 0.
          2. Zero scale_base  -> SiLU base contribution = 0.
          3. Keep scale_sp non-zero so gradients to coef flow during training.
          4. Run a calibration forward pass to measure any residual KAN output.
          5. Set theta_bias = inv_softplus(c) - kan_residual so that
             softplus(kan_residual + theta_bias) = softplus(inv_softplus(c)) = c.

        This compensates for any floating-point residual the KAN may still
        output after zeroing, giving a tighter warm-start match.
        """
        c = max(c, 1e-6)

        with torch.no_grad():
            # Step 1+2: zero all spline and base contributions
            for layer in self.kan.act_fun:
                if hasattr(layer, 'coef'):
                    layer.coef.data.zero_()
                if hasattr(layer, 'scale_base'):
                    layer.scale_base.data.zero_()

            if hasattr(self.kan, 'biases'):
                for b in self.kan.biases:
                    if hasattr(b, 'weight'):
                        b.weight.data.zero_()

            # Step 3: set bias to 0 first, then measure KAN residual
            self.theta_bias.data.zero_()

            # Step 4: calibration pass with a representative input
            # Use a typical residual magnitude (norm ~0.5 before log1p)
            in_dim = 1 if self.variant == '1norm' else 3
            dummy_feat = torch.zeros(1, in_dim, device=self.theta_bias.device)
            kan_residual = self.kan(dummy_feat).item()   # should be ~0

            # Step 5: theta_bias = inv_softplus(c) - kan_residual
            # inv_softplus(c) = log(exp(c) - 1)
            inv_sp = math.log(math.expm1(c))
            self.theta_bias.data.fill_(inv_sp - kan_residual)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        r     : (batch, M) residual
        theta : (batch, 1) predicted threshold, guaranteed > 0
        """
        features = self.compute_norms(r)                 # (batch, in_dim)
        features = torch.clamp(features, min=-2.0, max=4.0)  # stay within spline grid
        raw = self.kan(features)                          # (batch, 1)
        raw = raw + self.theta_bias                       # shift by learned bias
        raw = torch.clamp(raw, min=-10.0, max=5.0)        # Fix 4: prevent runaway
        theta = F.softplus(raw)                           # (batch, 1), > 0
        return theta
