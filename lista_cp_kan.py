"""
lista_cp_kan.py
LISTA-CP with KAN-based adaptive thresholds (LISTA-CP-KAN).

Drop-in replacement for LISTACPTorch where each fixed scalar theta^k is
replaced by a KANThetaPredictor that predicts a per-sample threshold from
the layer's residual norm statistics.

Only two changes from the base LISTA-CP:
  1. Constructor: 16 KANThetaPredictor modules instead of 16 scalar thetas.
  2. Forward: call predictor_k(res_k) to get (batch, 1) theta at each layer.

Forward pass at layer k (unchanged from LISTA-CP except theta source):
  res_k   = y - x_k @ A^T              (batch, M)
  theta_k = predictor_k(res_k)          (batch, 1)  <-- KAN output
  v_k     = x_k + res_k @ W_k^T        (batch, N)
  x_{k+1} = shrink_free(v_k, theta_k)
"""

import math
import numpy as np
import torch
import torch.nn as nn

from lista_cp_torch import shrink_free
from kan_theta_predictor import KANThetaPredictor


class LISTACPKan(nn.Module):
    """
    LISTA-CP-KAN: LISTA-CP with KAN adaptive thresholds.

    Parameters
    ----------
    A       : numpy array (M, N), the sensing matrix.
    T       : number of layers (default 16).
    lam     : initial lambda - used to derive W initialisation only.
    variant : '1norm' or '3norm' - selects KAN input features.
    """

    def __init__(self, A: np.ndarray, T: int = 16,
                 lam: float = 0.4, variant: str = '1norm',
                 kan_width: list = None):
        super().__init__()
        self._M, self._N = A.shape
        self._T = T
        self.variant = variant

        A32 = A.astype(np.float32)
        scale = float(1.001 * np.linalg.norm(A32, ord=2) ** 2)
        self._scale = scale

        self.register_buffer('A', torch.from_numpy(A32))

        W_init = (A32.T / scale).astype(np.float32)

        # Untied weight matrices W_k, one per layer, shape (N, M)
        self.Ws = nn.ParameterList([
            nn.Parameter(torch.from_numpy(W_init.copy()))
            for _ in range(T)
        ])

        # KAN threshold predictors - one per layer
        # kan_width allows overriding the default width for cross-version loading
        self.predictors = nn.ModuleList([
            KANThetaPredictor(variant=variant, M=self._M, width=kan_width)
            for _ in range(T)
        ])

    # ------------------------------------------------------------------
    def forward(self, y: torch.Tensor):
        """
        y    : (batch, M)
        Returns list [x0, x1, ..., xT], each (batch, N).
        """
        batch = y.size(0)
        xh = torch.zeros(batch, self._N, device=y.device, dtype=y.dtype)
        xhs = [xh]

        for k in range(self._T):
            W = self.Ws[k]                        # (N, M)

            res = y - xh @ self.A.T               # (batch, M)
            theta = self.predictors[k](res)        # (batch, 1)
            v = xh + res @ W.T                     # (batch, N)
            xh = shrink_free(v, theta)             # (batch, N), broadcasts theta
            xhs.append(xh)

        return xhs

    # ------------------------------------------------------------------
    def load_from_lista_cp(self, baseline):
        """
        Initialise this model from a trained LISTACPTorch baseline:
          - Copy all W_k weight matrices.
          - Warm-start each KAN predictor to output the corresponding theta^k.

        baseline : LISTACPTorch instance (already loaded from checkpoint).
        """
        with torch.no_grad():
            for k in range(self._T):
                # Copy W_k
                self.Ws[k].data.copy_(baseline.Ws[k].data)
                # Warm-start KAN predictor to output baseline theta^k
                theta_k = baseline.thetas[k].item()
                self.predictors[k].initialise_to_constant(theta_k)

        print("Loaded from LISTA-CP baseline.")
        print("  Warm-start thetas (initial KAN outputs ~=):",
              [round(baseline.thetas[k].item(), 5) for k in range(self._T)])

    # ------------------------------------------------------------------
    def get_kan_param_count(self):
        """Return total number of KAN-related trainable parameters."""
        total = 0
        for p in self.predictors.parameters():
            total += p.numel()
        return total

    def get_W_param_count(self):
        """Return total number of W weight parameters."""
        total = 0
        for p in self.Ws.parameters():
            total += p.numel()
        return total
