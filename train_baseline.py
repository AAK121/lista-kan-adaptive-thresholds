"""
train_baseline.py
Train the PyTorch LISTA-CP baseline and save lista_cp_baseline.pth.

Training faithfully reproduces the original LISTA-CP training scheme
(Chen et al., NeurIPS 2018) with 3 sub-stages per layer:

  For each layer t = 0..T-1:
    Sub-stage 1: Train ONLY new vars at layer t          lr = init_lr
    Sub-stage 2: Train ALL vars from layers 0..t         lr = init_lr * 0.2
    Sub-stage 3: Train ALL vars from layers 0..t         lr = init_lr * 0.02
  After completing layer t: multiply lr_multiplier of all trained vars by
  decay_rate=0.3 (so older layers see exponentially lower effective lr in
  future stages).

  Total: 16 * 3 = 48 stages. Each stage runs up to MAXIT steps and stops
  early if validation NMSE has not improved for BETTER_WAIT steps.

Loss matches original tf.nn.l2_loss: 0.5 * sum((xh - x)^2) over the batch.
Multi-GPU via DataParallel; batch size auto-scales with GPU count.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from problem_setup import (random_A, gen_samples_torch, load_test_data,
                           compute_nmse_db_torch)
from lista_cp_torch import LISTACPTorch

# -----------------------------------------------------------------------
M              = 250
N              = 500
T              = 16
PNZ            = 0.1
LAM            = 0.4
SNR            = np.inf

TBS            = 1024      # per-GPU batch size — large batch needed for tiny model to use GPU
VBS            = 1000      # validation batch size
VAL_STEP       = 100       # validate every this many steps
MAXIT          = 25000     # max steps (1024 * 25000 = 25.6M samples, 2x original 12.8M)
BETTER_WAIT    = 2000      # early-stop if no improvement for this many steps

INIT_LR        = 5e-4
LR_DECAY       = (0.2, 0.02)   # multipliers for sub-stages 2 and 3
DECAY_RATE     = 0.3            # lr multiplier applied to old vars after each layer

SAVE_PATH      = 'lista_cp_baseline.pth'

# On Kaggle set CHECKPOINT_DIR=/kaggle/working so files survive runtime resets.
# Files in /kaggle/working are committed when you hit "Save & Run All".
_DEFAULT_CKPT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', _DEFAULT_CKPT_DIR)
CHECKPOINT_EVERY_STEPS = int(os.environ.get('CHECKPOINT_EVERY_STEPS', '5000'))
XTEST_PATH     = 'LISTA-CPSS/data/xtest_n500_p10.npy'
SEED           = 42
# -----------------------------------------------------------------------


def get_base_model(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def make_fixed_val(A_t, n=1000, pnz=0.1, device='cpu'):
    y, x = gen_samples_torch(A_t, n, pnz=pnz, SNR=np.inf, device=device)
    return y.detach(), x.detach()


def checkpoint_path(filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)


def save_checkpoint(path, **payload):
    torch.save(payload, path)
    return path


def eval_layer_nmse(model, y_val, x_val):
    model.eval()
    with torch.no_grad():
        xhs  = model(y_val)
        nmse = [compute_nmse_db_torch(xh, x_val).item() for xh in xhs[1:]]
    model.train()
    return nmse


def l2_loss(xh, x):
    """
    Match original tf.nn.l2_loss(xh - x) = 0.5 * sum((xh-x)^2).
    Gradient direction is the same as MSE but unnormalized, matching
    the original TensorFlow implementation exactly.
    """
    return 0.5 * torch.sum((xh - x) ** 2)


# Pre-generate a pool of batches on GPU to reduce Python loop overhead.
# Each 'pool' holds POOL_SIZE batches; we cycle through them during training.
POOL_SIZE = 500   # batches in the pool; refreshed every POOL_SIZE steps

def make_batch_pool(A_t, batch_size, pnz, snr, device, n_batches):
    ys, xs = [], []
    for _ in range(n_batches):
        y, x = gen_samples_torch(A_t, batch_size, pnz=pnz, SNR=snr, device=device)
        ys.append(y)
        xs.append(x)
    return ys, xs


def run_substage(model, param_groups, A_t, pnz, snr, device,
                 y_val, x_val, stage_name, layer_idx,
                 lr_multiplier=None, use_grad_scaling=False,
                 checkpoint_name=None, tbs=64):
    """
    Run one sub-stage with early stopping.

    param_groups: list of dicts [{'params': [...], 'lr': float}, ...]
    Loss matches original: 0.5 * sum((xh - x)^2) over the batch.
    """
    optimizer = optim.Adam(param_groups)
    all_params = [p for g in param_groups for p in g['params']]

    nmse_hist   = []
    best_nmse   = float('inf')
    best_state  = None

    base = get_base_model(model)
    model.train()

    print(f"    {stage_name}", flush=True)

    # Pre-generate initial pool
    pool_y, pool_x = make_batch_pool(A_t, tbs, pnz, snr, device, POOL_SIZE)
    pool_idx = 0

    for step in range(1, MAXIT + 1):
        # Refresh pool every POOL_SIZE steps
        if pool_idx >= POOL_SIZE:
            pool_y, pool_x = make_batch_pool(A_t, tbs, pnz, snr, device, POOL_SIZE)
            pool_idx = 0

        y, x = pool_y[pool_idx], pool_x[pool_idx]
        pool_idx += 1

        optimizer.zero_grad()
        xhs  = model(y)
        loss = l2_loss(xhs[layer_idx + 1], x)
        loss.backward()

        if use_grad_scaling and lr_multiplier is not None:
            for p in all_params:
                if p.grad is None:
                    continue
                scale = lr_multiplier.get(id(p), 1.0)
                if scale != 1.0:
                    p.grad.data.mul_(scale)

        optimizer.step()

        if step % VAL_STEP == 0:
            model.eval()
            with torch.no_grad():
                xhs_v    = model(y_val)
                val_nmse = compute_nmse_db_torch(
                    xhs_v[layer_idx + 1], x_val).item()
            model.train()

            if val_nmse < best_nmse:
                best_nmse  = val_nmse
                best_state = {k: v.clone()
                              for k, v in base.state_dict().items()}
                if checkpoint_name is not None:
                    save_checkpoint(
                        checkpoint_path(checkpoint_name),
                        model_state_dict=best_state,
                        A=A_t.detach().cpu().numpy(),
                        best_nmse=best_nmse,
                        layer_idx=layer_idx,
                        stage_name=stage_name,
                    )

            nmse_hist.append(val_nmse)

            if checkpoint_name is not None and step % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(
                    checkpoint_path(
                        checkpoint_name.replace('.pth', '_latest.pth')),
                    model_state_dict=base.state_dict(),
                    A=A_t.detach().cpu().numpy(),
                    current_nmse=val_nmse,
                    best_nmse=best_nmse,
                    layer_idx=layer_idx,
                    stage_name=stage_name,
                    step=step,
                )

            if step % (100 * VAL_STEP) == 0:
                age_of_best = (len(nmse_hist) - int(np.argmin(nmse_hist)) - 1)
                print(f"      step {step:>7d}  val_nmse={val_nmse:.3f} dB  "
                      f"best={best_nmse:.3f} dB  "
                      f"age={age_of_best * VAL_STEP}", flush=True)
                if age_of_best * VAL_STEP > BETTER_WAIT:
                    print(f"      Early stop at step {step}", flush=True)
                    break

    if best_state is not None:
        base.load_state_dict(best_state)

    return best_nmse


def train_layer(model, layer_idx, A_t, pnz, snr, device,
                lr_multiplier, y_val, x_val, tbs):
    """
    Train one layer with the 3-sub-stage scheme from the original paper.
    """
    print(f"\n  Layer {layer_idx+1:2d}/{T}", flush=True)

    base = get_base_model(model)

    new_vars  = [base.Ws[layer_idx], base.thetas[layer_idx]]
    all_vars  = ([base.Ws[k] for k in range(layer_idx + 1)] +
                 [base.thetas[k] for k in range(layer_idx + 1)])

    for v in all_vars:
        v.requires_grad_(True)

    # Sub-stage 1: new vars only at init_lr
    run_substage(
        model,
        [{'params': new_vars, 'lr': INIT_LR}],
        A_t, pnz, snr, device, y_val, x_val,
        stage_name=(f"sub-stage 1  [layer {layer_idx+1} only, "
                    f"lr={INIT_LR:.2e}]"),
        layer_idx=layer_idx,
        lr_multiplier=lr_multiplier,
        use_grad_scaling=True,
        checkpoint_name=f'lista_cp_baseline_layer{layer_idx+1:02d}_sub1_best.pth',
        tbs=tbs,
    )

    # Sub-stages 2 and 3: all accumulated vars with decayed lr
    for sub_lr_factor in LR_DECAY:
        param_groups = [{'params': [v], 'lr': INIT_LR * sub_lr_factor}
                        for v in all_vars]
        base_lr = INIT_LR * sub_lr_factor
        run_substage(
            model,
            param_groups,
            A_t, pnz, snr, device, y_val, x_val,
            stage_name=(f"sub-stage fine-tune  [layers 0..{layer_idx+1}, "
                        f"base_lr={base_lr:.2e}]"),
            layer_idx=layer_idx,
            lr_multiplier=lr_multiplier,
            use_grad_scaling=True,
            checkpoint_name=(
                f'lista_cp_baseline_layer{layer_idx+1:02d}'
                f'_lr{int(sub_lr_factor*100):02d}_best.pth'
            ),
            tbs=tbs,
        )

    # Decay lr_multiplier for all trained vars
    for v in all_vars:
        vid = id(v)
        lr_multiplier[vid] = lr_multiplier.get(vid, 1.0) * DECAY_RATE

    layer_nmse = eval_layer_nmse(model, y_val, x_val)
    print(f"  Layer {layer_idx+1} done. val NMSE: "
          f"{[round(v, 2) for v in layer_nmse]}", flush=True)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # auto-tune CUDA kernels

    n_gpus = 1  # DataParallel disabled: model too small, sync overhead > gain
    tbs    = TBS

    print("=" * 65)
    print("LISTA-CP Baseline Training  (3-sub-stage greedy, T=16)")
    print(f"  M={M}, N={N}, T={T}, pnz={PNZ}")
    print(f"  MAXIT={MAXIT}, BETTER_WAIT={BETTER_WAIT}")
    print(f"  device={device}, n_gpus={n_gpus}, effective TBS={tbs}")
    print("=" * 65)

    A   = random_A(M, N, seed=SEED)
    A_t = torch.from_numpy(A).to(device)

    np.random.seed(SEED + 1)
    y_val, x_val = make_fixed_val(A_t, n=VBS, pnz=PNZ, device=device)
    print(f"Sensing matrix A: {A.shape},  val set: {y_val.shape[0]} samples")

    model = LISTACPTorch(A, T=T, lam=LAM).to(device)
    # DataParallel is skipped: model is tiny (~2M params), gradient sync overhead
    # between GPUs is larger than compute gain. Single GPU is faster here.
    if n_gpus > 1:
        print(f"  Note: {n_gpus} GPUs found but using 1 — model too small for DataParallel.")

    # torch.compile gives ~2x speedup on PyTorch >= 2.0 with no code changes.
    # Falls back silently if unavailable (older PyTorch or Windows).
    if hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception:
            pass

    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")

    lr_multiplier = {}

    # ------------------------------------------------------------------
    # Resume from the last completed per-layer checkpoint if it exists.
    # This lets training survive a Kaggle/Colab runtime reset.
    # ------------------------------------------------------------------
    start_layer = 0
    base = get_base_model(model)
    for layer_idx in range(T - 1, -1, -1):
        ckpt_fn = checkpoint_path(f'lista_cp_layer{layer_idx+1:02d}.pth')
        if os.path.exists(ckpt_fn):
            print(f"Resuming: found checkpoint for layer {layer_idx+1} "
                  f"at {ckpt_fn}")
            ckpt = torch.load(ckpt_fn, map_location=device, weights_only=False)
            base.load_state_dict(ckpt['model_state_dict'])
            start_layer = layer_idx + 1
            # Reconstruct lr_multiplier state: every var in layers 0..layer_idx
            # has been decayed (layer_idx+1) times.
            for k in range(start_layer):
                decay_k = DECAY_RATE ** (start_layer - k)
                for v in [base.Ws[k], base.thetas[k]]:
                    lr_multiplier[id(v)] = decay_k
            if start_layer < T:
                print(f"  Loaded weights. Resuming from layer {start_layer+1}.")
            else:
                print(f"  All {T} layers already completed. Skipping training.")
            print(f"  lr_multiplier entries restored: {len(lr_multiplier)}")
            break

    if start_layer == 0:
        print("No previous checkpoint found. Training from scratch.")

    t0 = time.time()

    for layer in range(start_layer, T):
        train_layer(model, layer, A_t, PNZ, SNR, device,
                    lr_multiplier, y_val, x_val, tbs)

        base = get_base_model(model)
        save_checkpoint(
            checkpoint_path(f'lista_cp_layer{layer+1:02d}.pth'),
            model_state_dict=base.state_dict(),
            A=A,
            config=dict(M=M, N=N, T=T, lam=LAM, pnz=PNZ),
            layer_completed=layer + 1,
        )

    print(f"\nTotal training time: {(time.time()-t0)/60:.1f} min")

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    base = get_base_model(model)
    if os.path.exists(XTEST_PATH):
        y_test, x_test = load_test_data(XTEST_PATH, A, SNR=np.inf)
        y_t = torch.from_numpy(y_test).to(device)
        x_t = torch.from_numpy(x_test).to(device)
    else:
        print("  Test file not found; using validation set.")
        y_t, x_t = y_val, x_val

    layer_nmse = eval_layer_nmse(model, y_t, x_t)
    print("\nPer-layer NMSE on test set:")
    for k, v in enumerate(layer_nmse):
        print(f"  Layer {k+1:2d}: {v:.2f} dB")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    thetas = [t.item() for t in base.thetas]
    save_checkpoint(
        checkpoint_path(SAVE_PATH),
        model_state_dict=base.state_dict(),
        A=A,
        thetas=thetas,
        scale=base._scale,
        config=dict(M=M, N=N, T=T, lam=LAM, pnz=PNZ),
        test_nmse_per_layer=layer_nmse,
    )
    print(f"\nCheckpoint saved to '{checkpoint_path(SAVE_PATH)}'")
    print(f"Final NMSE layer {T}: {layer_nmse[-1]:.2f} dB")
    print(f"Trained thetas: {[round(v, 5) for v in thetas]}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
