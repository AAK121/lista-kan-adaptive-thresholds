"""
lista_cp_torch.py
PyTorch reimplementation of LISTA-CP (weight-coupled LISTA).

Reference: Chen et al., "Theoretical Linear Convergence of Unfolded ISTA and
its Practical Weights and Thresholds", NeurIPS 2018.

Convention: batch-first tensors
  x: (batch, N)   sparse codes
  y: (batch, M)   measurements
  A: (M, N)       sensing matrix (fixed)
  W_k: (N, M)     learned weight matrix at layer k  [initialised to A^T/scale]
  theta_k: scalar learned threshold at layer k      [initialised to lam/scale]

Forward pass at layer k:
  res_k   = y - x_k @ A^T              (batch, M)
  v_k     = x_k + res_k @ W_k^T        (batch, N)
  x_{k+1} = shrink_free(v_k, theta_k)
"""

import numpy as np
import torch
import torch.nn as nn


def shrink_free(v, theta):
    """
    Soft thresholding without positivity constraint on theta.
    v:     (..., N)
    theta: scalar, or (batch, 1) for per-sample thresholds
    """
    return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0.0)


class LISTACPTorch(nn.Module):
    """
    LISTA-CP with untied weights.
    All T layers share the same A (fixed), but each has its own W_k and theta_k.
    """

    def __init__(self, A: np.ndarray, T: int = 16, lam: float = 0.4):
        """
        A   : numpy array (M, N), the sensing matrix.
        T   : number of layers.
        lam : initial lambda — used to derive initial threshold.
        """
        super().__init__()
        self._M, self._N = A.shape
        self._T = T

        A32 = A.astype(np.float32)
        scale = float(1.001 * np.linalg.norm(A32, ord=2) ** 2)
        self._scale = scale

        # Fixed sensing matrix (not a parameter)
        self.register_buffer('A', torch.from_numpy(A32))

        # Initial weight: W = A^T / scale  →  shape (N, M)
        W_init = (A32.T / scale).astype(np.float32)
        theta_init = float(lam / scale)

        # Untied: one W and one theta per layer
        self.Ws = nn.ParameterList([
            nn.Parameter(torch.from_numpy(W_init.copy()))
            for _ in range(T)
        ])
        self.thetas = nn.ParameterList([
            nn.Parameter(torch.tensor(theta_init))
            for _ in range(T)
        ])

    # ------------------------------------------------------------------
    def forward(self, y):
        """
        y: (batch, M)
        Returns list of length T+1: [x0, x1, ..., xT], each (batch, N).
        x0 is the zero initialisation.
        """
        batch = y.size(0)
        xh = torch.zeros(batch, self._N, device=y.device, dtype=y.dtype)
        xhs = [xh]

        for k in range(self._T):
            W = self.Ws[k]        # (N, M)
            theta = self.thetas[k]  # scalar

            res = y - xh @ self.A.T   # (batch, M)
            v = xh + res @ W.T         # (batch, N)
            xh = shrink_free(v, theta)
            xhs.append(xh)

        return xhs

    # ------------------------------------------------------------------
    def get_theta_values(self):
        """Return list of scalar theta values (detached numpy floats)."""
        return [t.item() for t in self.thetas]

    def get_W_tensors(self):
        """Return list of W parameter tensors (detached copies)."""
        return [w.detach().clone() for w in self.Ws]
