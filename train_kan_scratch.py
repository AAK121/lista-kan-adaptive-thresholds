"""
train_kan_scratch.py
Train LISTA-CP-KAN from scratch using the same greedy layer-by-layer
scheme as the baseline (3 sub-stages per layer).

This lets W and KAN thresholds co-evolve together from random init,
avoiding the fine-tuning degradation problem.

For each layer t = 0..T-1:
  Sub-stage 1: Train ONLY new vars (W_t + predictor_t)    lr = init_lr
  Sub-stage 2: Train ALL vars from layers 0..t            lr = init_lr * 0.2
  Sub-stage 3: Train ALL vars from layers 0..t            lr = init_lr * 0.02
  After completing layer t: decay lr_multiplier by 0.3.

Usage:
  python train_kan_scratch.py --variant 1norm
  python train_kan_scratch.py --variant 3norm
"""

import os, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from problem_setup import (random_A, gen_samples_torch, load_test_data,
                           compute_nmse_db_torch)
from lista_cp_kan import LISTACPKan

# -----------------------------------------------------------------------
M              = 250
N              = 500
T              = 16
PNZ            = 0.1
LAM            = 0.4
SNR            = np.inf

TBS            = 1024
VBS            = 1000
VAL_STEP       = 100
MAXIT          = 25000
BETTER_WAIT    = 2000

INIT_LR        = 5e-4
LR_DECAY       = (0.2, 0.02)
DECAY_RATE     = 0.3

_DEFAULT_CKPT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', _DEFAULT_CKPT_DIR)
CHECKPOINT_EVERY_STEPS = int(os.environ.get('CHECKPOINT_EVERY_STEPS', '5000'))
XTEST_PATH     = 'LISTA-CPSS/data/xtest_n500_p10.npy'
SEED           = 42
POOL_SIZE      = 500
GRAD_CLIP      = 1.0
REG_LAMBDA     = 1e-4
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
    return 0.5 * torch.sum((xh - x) ** 2)


def kan_spline_l1(model):
    base = get_base_model(model)
    reg = torch.tensor(0.0, device=next(base.parameters()).device)
    for predictor in base.predictors:
        for layer in predictor.kan.act_fun:
            if hasattr(layer, 'coef'):
                reg = reg + layer.coef.abs().mean()
    return reg


def make_batch_pool(A_t, batch_size, pnz, snr, device, n_batches):
    ys, xs = [], []
    for _ in range(n_batches):
        y, x = gen_samples_torch(A_t, batch_size, pnz=pnz, SNR=snr, device=device)
        ys.append(y)
        xs.append(x)
    return ys, xs


def get_new_vars(base, layer_idx):
    """Get trainable params for the new layer only: W_k + predictor_k."""
    params = [base.Ws[layer_idx]]
    params += list(base.predictors[layer_idx].parameters())
    return params


def get_all_vars(base, layer_idx):
    """Get all trainable params for layers 0..layer_idx: all Ws + all predictors."""
    params = []
    for k in range(layer_idx + 1):
        params.append(base.Ws[k])
        params += list(base.predictors[k].parameters())
    return params


def run_substage(model, param_groups, A_t, pnz, snr, device,
                 y_val, x_val, stage_name, layer_idx,
                 lr_multiplier=None, use_grad_scaling=False,
                 checkpoint_name=None, tbs=1024):
    optimizer = optim.Adam(param_groups)
    all_params = [p for g in param_groups for p in g['params']]

    nmse_hist   = []
    best_nmse   = float('inf')
    best_state  = None

    base = get_base_model(model)
    model.train()

    print(f"    {stage_name}", flush=True)

    pool_y, pool_x = make_batch_pool(A_t, tbs, pnz, snr, device, POOL_SIZE)
    pool_idx = 0

    for step in range(1, MAXIT + 1):
        if pool_idx >= POOL_SIZE:
            pool_y, pool_x = make_batch_pool(A_t, tbs, pnz, snr, device, POOL_SIZE)
            pool_idx = 0

        y, x = pool_y[pool_idx], pool_x[pool_idx]
        pool_idx += 1

        optimizer.zero_grad()
        xhs  = model(y)
        loss = l2_loss(xhs[layer_idx + 1], x)
        loss = loss + REG_LAMBDA * kan_spline_l1(model)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP)

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
                    )

            nmse_hist.append(val_nmse)

            if step % (10 * VAL_STEP) == 0:
                age_of_best = (len(nmse_hist) - int(np.argmin(nmse_hist)) - 1)
                print(f"      step {step:>7d}  val_nmse={val_nmse:.3f} dB  "
                      f"best={best_nmse:.3f} dB  "
                      f"age={age_of_best * VAL_STEP}", flush=True)

            age_of_best = (len(nmse_hist) - int(np.argmin(nmse_hist)) - 1)
            if age_of_best * VAL_STEP >= BETTER_WAIT:
                print(f"      Early stop at step {step}", flush=True)
                break

    if best_state is not None:
        base.load_state_dict(best_state)

    return best_nmse


def train_layer(model, layer_idx, A_t, pnz, snr, device,
                lr_multiplier, y_val, x_val, tbs):
    print(f"\n  Layer {layer_idx+1:2d}/{T}", flush=True)

    base = get_base_model(model)

    new_vars = get_new_vars(base, layer_idx)
    all_vars = get_all_vars(base, layer_idx)

    for v in all_vars:
        v.requires_grad_(True)

    # Sub-stage 1: new layer vars only at init_lr
    run_substage(
        model,
        [{'params': new_vars, 'lr': INIT_LR}],
        A_t, pnz, snr, device, y_val, x_val,
        stage_name=(f"sub-stage 1  [layer {layer_idx+1} only, "
                    f"lr={INIT_LR:.2e}]"),
        layer_idx=layer_idx,
        lr_multiplier=lr_multiplier,
        use_grad_scaling=True,
        checkpoint_name=f'lista_cp_kan_scratch_layer{layer_idx+1:02d}_sub1_best.pth',
        tbs=tbs,
    )

    # Sub-stages 2 and 3: all vars with decayed lr
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
                f'lista_cp_kan_scratch_layer{layer_idx+1:02d}'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='3norm',
                        choices=['1norm', '3norm'])
    args = parser.parse_args()
    variant = args.variant

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    tbs = TBS
    tag = '1' if variant == '1norm' else '3'
    save_name = f'lista_cp_kan{tag}_scratch.pth'

    print("=" * 65)
    print(f"LISTA-CP-KAN From-Scratch Training  (variant={variant}, T={T})")
    print(f"  M={M}, N={N}, T={T}, pnz={PNZ}")
    print(f"  MAXIT={MAXIT}, BETTER_WAIT={BETTER_WAIT}")
    print(f"  device={device}, TBS={tbs}")
    print(f"  GRAD_CLIP={GRAD_CLIP}, REG_LAMBDA={REG_LAMBDA}")
    print("=" * 65)

    A   = random_A(M, N, seed=SEED)
    A_t = torch.from_numpy(A).to(device)

    np.random.seed(SEED + 1)
    y_val, x_val = make_fixed_val(A_t, n=VBS, pnz=PNZ, device=device)
    print(f"Sensing matrix A: {A.shape},  val set: {y_val.shape[0]} samples")

    model = LISTACPKan(A, T=T, lam=LAM, variant=variant).to(device)

    total = sum(p.numel() for p in model.parameters())
    w_params = model.get_W_param_count()
    kan_params = model.get_kan_param_count()
    print(f"Parameters: W={w_params:,}  KAN={kan_params:,}  Total={total:,}")

    lr_multiplier = {}

    # ------------------------------------------------------------------
    # Resume from last completed layer checkpoint
    # ------------------------------------------------------------------
    start_layer = 0
    base = get_base_model(model)
    for layer_idx in range(T - 1, -1, -1):
        ckpt_fn = checkpoint_path(f'lista_cp_kan_scratch_layer{layer_idx+1:02d}.pth')
        if os.path.exists(ckpt_fn):
            print(f"Resuming: found checkpoint for layer {layer_idx+1}")
            ckpt = torch.load(ckpt_fn, map_location=device, weights_only=False)
            base.load_state_dict(ckpt['model_state_dict'])
            start_layer = layer_idx + 1
            for k in range(start_layer):
                decay_k = DECAY_RATE ** (start_layer - k)
                params_k = [base.Ws[k]] + list(base.predictors[k].parameters())
                for v in params_k:
                    lr_multiplier[id(v)] = decay_k
            if start_layer < T:
                print(f"  Resuming from layer {start_layer+1}.")
            else:
                print(f"  All {T} layers completed. Skipping training.")
            print(f"  lr_multiplier entries: {len(lr_multiplier)}")
            break

    if start_layer == 0:
        print("Training from scratch.")

    t0 = time.time()

    for layer in range(start_layer, T):
        train_layer(model, layer, A_t, PNZ, SNR, device,
                    lr_multiplier, y_val, x_val, tbs)

        base = get_base_model(model)
        save_checkpoint(
            checkpoint_path(f'lista_cp_kan_scratch_layer{layer+1:02d}.pth'),
            model_state_dict=base.state_dict(),
            A=A,
            config=dict(M=M, N=N, T=T, lam=LAM, pnz=PNZ),
            variant=variant,
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

    save_checkpoint(
        checkpoint_path(save_name),
        model_state_dict=base.state_dict(),
        A=A,
        config=dict(M=M, N=N, T=T, lam=LAM, pnz=PNZ),
        variant=variant,
        test_nmse_per_layer=layer_nmse,
    )
    print(f"\nCheckpoint saved to '{checkpoint_path(save_name)}'")
    print(f"Final NMSE layer {T}: {layer_nmse[-1]:.2f} dB")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
