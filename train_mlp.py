"""
train_mlp.py
Joint fine-tuning of LISTA-CP-MLP (MLP per-element adaptive thresholds).

Strategy:
  1. Load trained LISTA-CP baseline.
  2. Warm-start MLP predictors to match baseline thetas exactly.
  3. Joint fine-tuning: W_k at very low LR, MLP predictors at higher LR.
  4. Per-batch NaN skip + gradient clipping for stability.

Usage:
  python train_mlp.py
"""

import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from problem_setup import (gen_samples_torch, load_test_data,
                           compute_nmse_db_torch)
from lista_cp_torch import LISTACPTorch
from lista_cp_mlp import LISTACPMlp

# -----------------------------------------------------------------------
BASELINE_PATH  = 'lista_cp_baseline.pth'
XTEST_PATH     = 'LISTA-CPSS/data/xtest_n500_p10.npy'
_DEFAULT_CKPT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', _DEFAULT_CKPT_DIR)

EPOCHS             = 100
BATCHES_PER_EPOCH  = 100
BATCH_SIZE         = 1024

LR_W               = 1e-6   # very conservative — W already well-trained
LR_MLP             = 1e-4   # MLP can handle higher LR than KAN splines
GRAD_CLIP          = 1.0
HIDDEN             = 128

SEED = 42
# -----------------------------------------------------------------------


def checkpoint_path(filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)


def load_baseline(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    A     = ckpt['A']
    cfg   = ckpt['config']
    model = LISTACPTorch(A, T=cfg['T'], lam=cfg['lam']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Baseline loaded: {path!r}")
    print(f"  Baseline NMSE per layer: "
          f"{[round(v, 2) for v in ckpt.get('test_nmse_per_layer', [])]}")
    return model, A, cfg


def make_val_set(A_t, n=1000, pnz=0.1, device='cpu'):
    y, x = gen_samples_torch(A_t, n, pnz=pnz, SNR=np.inf, device=device)
    return y.detach(), x.detach()


def eval_nmse(model, y_val, x_val):
    model.eval()
    with torch.no_grad():
        xhs  = model(y_val)
        nmse = compute_nmse_db_torch(xhs[-1], x_val).item()
    model.train()
    return nmse


def eval_layer_nmse(model, y_val, x_val):
    model.eval()
    with torch.no_grad():
        xhs  = model(y_val)
        nmse = [compute_nmse_db_torch(xh, x_val).item() for xh in xhs[1:]]
    model.train()
    return nmse


def nmse_loss(xh, x):
    return torch.mean(
        torch.sum((xh - x) ** 2, dim=1) / (torch.sum(x ** 2, dim=1) + 1e-10)
    )


def prefetch(A_t, n, bs, device):
    ys, xs = [], []
    for _ in range(n):
        y, x = gen_samples_torch(A_t, bs, pnz=0.1, SNR=np.inf, device=device)
        ys.append(y); xs.append(x)
    return ys, xs


def train(model, A_t, device, y_val, x_val):
    T = model._T

    for w in model.Ws:
        w.requires_grad_(True)
    for p in model.predictors.parameters():
        p.requires_grad_(True)

    optimizer = optim.Adam([
        {'params': list(model.Ws.parameters()),         'lr': LR_W},
        {'params': list(model.predictors.parameters()), 'lr': LR_MLP},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-8)

    use_amp = (device == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]

    best_nmse  = float('inf')
    best_state = None

    print(f"\n{'='*55}")
    print(f"  Joint fine-tuning  ({EPOCHS} epochs x {BATCHES_PER_EPOCH} batches)")
    print(f"  LR_W={LR_W:.0e}  LR_MLP={LR_MLP:.0e}  BS={BATCH_SIZE}")
    print(f"{'='*55}")

    for epoch in range(1, EPOCHS + 1):
        pool_y, pool_x = prefetch(A_t, BATCHES_PER_EPOCH, BATCH_SIZE, device)
        epoch_loss  = 0.0
        valid_batches = 0

        for y, x in zip(pool_y, pool_x):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', enabled=use_amp):
                xhs  = model(y)
                loss = sum(nmse_loss(xh, x) for xh in xhs[1:]) / T
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss  += loss.item()
            valid_batches += 1

        if valid_batches == 0:
            print(f"  All batches NaN at epoch {epoch} — stopping.")
            break

        scheduler.step()
        avg_loss = epoch_loss / valid_batches
        val_nmse = eval_nmse(model, y_val, x_val)

        is_best = val_nmse < best_nmse
        if is_best:
            best_nmse  = val_nmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({'model_state_dict': best_state,
                        'best_nmse': best_nmse, 'epoch': epoch},
                       checkpoint_path('lista_cp_mlp_best.pth'))

        suffix = "  [best]" if is_best else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"avg_loss={avg_loss:.4f}  val_nmse={val_nmse:.2f} dB{suffix}",
              flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best: {best_nmse:.2f} dB")

    return model


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # Skip if already done
    final_path = checkpoint_path('lista_cp_mlp_final.pth')
    if os.path.exists(final_path):
        print(f"Final checkpoint found at {final_path!r}. Skipping training.")
        return

    baseline_path = checkpoint_path(BASELINE_PATH)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline not found: {baseline_path!r}. "
                                f"Run train_baseline.py first.")

    baseline_model, A, cfg = load_baseline(baseline_path, device)
    A_t = torch.from_numpy(A).to(device)

    np.random.seed(SEED + 1)
    y_val, x_val = make_val_set(A_t, n=1000, device=device)

    model = LISTACPMlp(A, T=cfg['T'], lam=cfg.get('lam', 0.4),
                       hidden=HIDDEN).to(device)
    model.load_from_lista_cp(baseline_model)

    init_nmse = eval_nmse(model, y_val, x_val)
    print(f"Warm-start NMSE: {init_nmse:.2f} dB  (baseline reference)")

    w_params   = model.get_W_param_count()
    mlp_params = model.get_mlp_param_count()
    print(f"Parameters — W: {w_params:,}  MLP: {mlp_params:,}  "
          f"Total: {w_params + mlp_params:,}")

    t0 = time.time()
    model = train(model, A_t, device, y_val, x_val)

    # Final evaluation
    if os.path.exists(XTEST_PATH):
        y_test, x_test = load_test_data(XTEST_PATH, A, SNR=np.inf)
        y_t = torch.from_numpy(y_test).to(device)
        x_t = torch.from_numpy(x_test).to(device)
    else:
        print("  Test file not found; using val set.")
        y_t, x_t = y_val, x_val

    layer_nmse = eval_layer_nmse(model, y_t, x_t)
    print("\nPer-layer NMSE on test set:")
    for k, v in enumerate(layer_nmse):
        print(f"  Layer {k+1:2d}: {v:.2f} dB")

    torch.save({
        'model_state_dict': model.state_dict(),
        'A': A,
        'config': cfg,
        'hidden': HIDDEN,
        'test_nmse_per_layer': layer_nmse,
    }, final_path)
    print(f"\nSaved to {final_path!r}")
    print(f"Final NMSE layer {cfg['T']}: {layer_nmse[-1]:.2f} dB")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
