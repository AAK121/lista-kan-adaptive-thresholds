"""
lista_cp_mlp.py
LISTA-CP with MLP-based per-element adaptive thresholds (LISTA-CP-MLP).

Drop-in replacement for LISTACPTorch where each fixed scalar theta^k is
replaced by an MLPThetaPredictor that outputs a per-element threshold
from the pre-threshold signal v_k.

Forward pass at layer k:
  res_k   = y - x_k @ A^T              (batch, M)
  v_k     = x_k + res_k @ W_k^T        (batch, N)
  theta_k = predictor_k(v_k)            (batch, N)  <-- MLP per-element
  x_{k+1} = shrink_free(v_k, theta_k)   (batch, N)
"""

import numpy as np
import torch
import torch.nn as nn

from lista_cp_torch import shrink_free
from mlp_theta_predictor import MLPThetaPredictor


class LISTACPMlp(nn.Module):
    """
    LISTA-CP-MLP: LISTA-CP with MLP per-element adaptive thresholds.

    Parameters
    ----------
    A      : numpy array (M, N), the sensing matrix.
    T      : number of layers (default 16).
    lam    : initial lambda - used to derive W initialisation only.
    hidden : hidden layer size of each MLPThetaPredictor (default 128).
    """

    def __init__(self, A: np.ndarray, T: int = 16,
                 lam: float = 0.4, hidden: int = 128):
        super().__init__()
        self._M, self._N = A.shape
        self._T = T

        A32 = A.astype(np.float32)
        scale = float(1.001 * np.linalg.norm(A32, ord=2) ** 2)
        self._scale = scale

        self.register_buffer('A', torch.from_numpy(A32))

        W_init = (A32.T / scale).astype(np.float32)

        self.Ws = nn.ParameterList([
            nn.Parameter(torch.from_numpy(W_init.copy()))
            for _ in range(T)
        ])

        self.predictors = nn.ModuleList([
            MLPThetaPredictor(N=self._N, hidden=hidden)
            for _ in range(T)
        ])

    def forward(self, y: torch.Tensor):
        """
        y    : (batch, M)
        Returns list [x0, x1, ..., xT], each (batch, N).
        """
        batch = y.size(0)
        xh = torch.zeros(batch, self._N, device=y.device, dtype=y.dtype)
        xhs = [xh]

        for k in range(self._T):
            W   = self.Ws[k]                          # (N, M)
            res = y - xh @ self.A.T                   # (batch, M)
            v   = xh + res @ W.T                      # (batch, N)
            theta = self.predictors[k](v)              # (batch, N)
            xh  = shrink_free(v, theta)               # (batch, N)
            xhs.append(xh)

        return xhs

    def load_from_lista_cp(self, baseline):
        """
        Initialise from a trained LISTACPTorch baseline:
          - Copy all W_k weight matrices.
          - Warm-start each MLP predictor to output the baseline theta^k.
        """
        with torch.no_grad():
            for k in range(self._T):
                self.Ws[k].data.copy_(baseline.Ws[k].data)
                theta_k = baseline.thetas[k].item()
                self.predictors[k].initialise_to_constant(theta_k)

        print("Loaded from LISTA-CP baseline.")
        print("  Warm-start thetas:",
              [round(baseline.thetas[k].item(), 5) for k in range(self._T)])

    def get_mlp_param_count(self):
        return sum(p.numel() for p in self.predictors.parameters())

    def get_W_param_count(self):
        return sum(p.numel() for p in self.Ws.parameters())
