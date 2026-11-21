"""
problem_setup.py
Data generation and problem utilities for LISTA-CP experiments.

Convention throughout: batch-first tensors
  x: (batch, N)  sparse codes
  y: (batch, M)  measurements
  A: (M, N)      sensing matrix
"""

import os
import numpy as np
import numpy.linalg as la
import torch


def random_A(M, N, con_num=0.0, col_normalized=True, seed=None):
    """Generate a random sensing matrix of shape (M, N)."""
    rng = np.random.RandomState(seed)
    A = rng.normal(scale=1.0 / np.sqrt(M), size=(M, N)).astype(np.float32)
    if con_num > 0:
        U, _, V = la.svd(A, full_matrices=False)
        s = np.logspace(0, np.log10(1.0 / con_num), M)
        A = (U * (s * np.sqrt(N) / la.norm(s))) @ V
        A = A.astype(np.float32)
    if col_normalized:
        A = A / np.sqrt(np.sum(np.square(A), axis=0, keepdims=True))
    return A


def gen_samples_np(A, batch_size, pnz=0.1, SNR=np.inf):
    """
    Generate (y, x) pairs as numpy arrays, shapes (batch, M) and (batch, N).
    """
    M, N = A.shape
    # Bernoulli support mask
    support = (np.random.uniform(size=(batch_size, N)) <= pnz).astype(np.float32)
    # Gaussian magnitudes
    x = support * np.random.normal(size=(batch_size, N)).astype(np.float32)
    # Noiseless measurement
    y = (A @ x.T).T  # (batch, M)
    # Add noise if SNR is finite
    if np.isfinite(SNR):
        std = np.std(y, axis=1, keepdims=True) * np.power(10.0, -SNR / 20.0)
        std = np.maximum(std, 1e-50)
        y = y + np.random.normal(size=y.shape).astype(np.float32) * std
    return y.astype(np.float32), x.astype(np.float32)


def gen_samples_torch(A_t, batch_size, pnz=0.1, SNR=np.inf, device='cpu'):
    """
    Generate (y, x) pairs as torch tensors.
    A_t: tensor (M, N)
    returns y (batch, M), x (batch, N)
    """
    M, N = A_t.shape
    support = (torch.rand(batch_size, N, device=device) <= pnz).float()
    x = support * torch.randn(batch_size, N, device=device)
    y = x @ A_t.T  # (batch, M)
    if np.isfinite(SNR):
        std = y.std(dim=1, keepdim=True) * (10.0 ** (-SNR / 20.0))
        std = torch.clamp(std, min=1e-50)
        y = y + torch.randn_like(y) * std
    return y, x


def load_test_data(xtest_path, A, SNR=np.inf):
    """
    Load test x from .npy file (original shape N×1000, column-major),
    transpose to (1000, N), compute y = Ax + noise.
    Returns y_test (1000, M), x_test (1000, N) as float32 numpy arrays.
    """
    xt = np.load(xtest_path).astype(np.float32)
    if xt.ndim == 2 and xt.shape[0] != xt.shape[1]:
        # original file is (N, num_samples) — transpose to (num_samples, N)
        if xt.shape[0] == A.shape[1]:
            xt = xt.T  # now (num_samples, N)
    x_test = xt  # (num_samples, N)
    y_test = (A @ x_test.T).T  # (num_samples, M)
    if np.isfinite(SNR):
        std = np.std(y_test, axis=1, keepdims=True) * np.power(10.0, -SNR / 20.0)
        std = np.maximum(std, 1e-50)
        y_test = y_test + np.random.normal(size=y_test.shape).astype(np.float32) * std
    return y_test.astype(np.float32), x_test.astype(np.float32)


def compute_nmse_db(xh, x):
    """
    NMSE in dB.
    xh, x: numpy arrays of same shape (batch, N).
    """
    num = np.sum(np.square(xh - x))
    den = np.sum(np.square(x)) + 1e-10
    return 10.0 * np.log10(num / den)


def compute_nmse_db_torch(xh, x):
    """NMSE in dB, torch tensors."""
    num = torch.sum((xh - x) ** 2)
    den = torch.sum(x ** 2) + 1e-10
    return 10.0 * torch.log10(num / den)
