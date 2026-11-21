"""
mlp_theta_predictor.py
MLP-based per-element adaptive threshold predictor for LISTA-CP-MLP.

Input  : v_k  (batch, N) — pre-threshold signal at layer k
Output : θ_k  (batch, N) — per-element threshold, guaranteed > 0

Architecture:
  v_k → Linear(N, hidden) → ReLU → Linear(hidden, N) → + bias → clamp → softplus → θ_k

warm-start: zero all weights, set bias = inv_softplus(c) so every element
outputs exactly c (matching the baseline scalar theta) at init.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPThetaPredictor(nn.Module):
    def __init__(self, N: int = 500, hidden: int = 128):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.LayerNorm(N),
            nn.Linear(N, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N),
        )
        # Per-element bias after MLP; set by initialise_to_constant
        self.theta_bias = nn.Parameter(torch.zeros(N))

    def initialise_to_constant(self, c: float):
        """
        Configure so forward() returns c for every element and any input.
        Zeros all Linear weights/biases so the MLP output is 0 for any input.
        LayerNorm is left at default (weight=1, bias=0) — it normalises but
        the downstream Linear has zero weights so its output is still 0.
        Then sets theta_bias = inv_softplus(c).
        """
        c = max(c, 1e-6)
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.zero_()
                    layer.bias.data.zero_()
            inv_sp = math.log(math.expm1(c))
            self.theta_bias.data.fill_(inv_sp)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v     : (batch, N)
        theta : (batch, N), per-element threshold > 0
        """
        raw = self.net(v) + self.theta_bias          # (batch, N)
        raw = torch.clamp(raw, min=-10.0, max=5.0)   # prevent runaway
        return F.softplus(raw)                        # (batch, N), > 0
