"""
train_kan.py
Two-phase KAN training for both '1norm' and '3norm' variants.

Phase A - KAN only, W^k frozen
  Warmup  (5  epochs): KAN lr=1e-7 -- splines stabilise from constant init.
  Main    (0  epochs): skipped -- W-frozen KAN training breaks W/theta coupling.

Phase B - Joint fine-tuning (100 epochs)
  Unfreeze W_k.
  Two-group Adam: W_k at lr=1e-5, KAN predictors at lr=1e-5.
  After warmup confirms init, joint training lets W and KAN co-adapt.
  Grad-clip max-norm 1.0 on all parameters.

After warmup, KAN spline grids are updated to match the actual
residual-norm distribution seen during inference, giving the splines
better resolution where the inputs actually land.

Checkpoints saved as:
  lista_cp_kan1_final.pth  (1norm variant)
  lista_cp_kan3_final.pth  (3norm variant)

Usage:
  python train_kan.py --variant 1norm
  python train_kan.py --variant 3norm
  python train_kan.py --variant both   (runs 1norm then 3norm)
"""

import os, math, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from problem_setup import (gen_samples_torch, load_test_data,
                           compute_nmse_db_torch)
from lista_cp_torch import LISTACPTorch
from lista_cp_kan import LISTACPKan
from lista_cp_torch import shrink_free

# -----------------------------------------------------------------------
BASELINE_PATH  = 'lista_cp_baseline.pth'
XTEST_PATH     = 'LISTA-CPSS/data/xtest_n500_p10.npy'
_DEFAULT_CKPT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', _DEFAULT_CKPT_DIR)
CHECKPOINT_EVERY_EPOCHS = int(os.environ.get('CHECKPOINT_EVERY_EPOCHS', '5'))

PHASE_A_WARMUP_EPOCHS = 5    # gentle warmup -- confirms KAN init matches baseline
PHASE_A_MAIN_EPOCHS   = 0    # skipped -- W-frozen training breaks W/theta coupling
PHASE_B_EPOCHS        = 100  # joint fine-tuning
BATCHES_PER_EPOCH     = 100  # fewer batches per epoch since batch is larger
BATCH_SIZE            = 1024 # large batch needed for tiny model to actually use GPU

LR_KAN_WARMUP = 1e-7   # warmup lr -- splines barely move
LR_KAN        = 1e-6   # Phase B KAN lr -- lowered for deeper [3,6,1] / [1,2,1] networks
LR_W          = 1e-6   # Phase B W lr -- very conservative; W is already well-trained
GRAD_CLIP     = 1.0
KAN_WEIGHT_DECAY = 1e-4  # weight decay on KAN params to prevent unbounded growth
REG_LAMBDA    = 1e-3     # L1 regularization on spline coefs

SEED = 123
# -----------------------------------------------------------------------


def get_base_model(model):
    """Unwrap DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def checkpoint_path(filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)


def save_checkpoint(path, **payload):
    torch.save(payload, path)
    return path


def load_baseline(path: str, device: str):
    """Load baseline checkpoint, reconstruct LISTACPTorch, return (model, A, cfg)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    A    = ckpt['A']
    cfg  = ckpt['config']
    model = LISTACPTorch(A, T=cfg['T'], lam=cfg['lam']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Baseline loaded from {path!r}")
    print(f"  Baseline NMSE per layer: "
          f"{[round(v, 2) for v in ckpt.get('test_nmse_per_layer', [])]}")
    return model, A, cfg


def make_val_set(A_t, n=1000, pnz=0.1, snr=np.inf, device='cpu'):
    y, x = gen_samples_torch(A_t, n, pnz=pnz, SNR=snr, device=device)
    return y.detach(), x.detach()


def eval_nmse(model, y_val, x_val):
    model.eval()
    with torch.no_grad():
        xhs = model(y_val)
        nmse = compute_nmse_db_torch(xhs[-1], x_val).item()
    model.train()
    return nmse


def eval_layer_nmse(model, y_val, x_val):
    """Return list of NMSE (dB) at each layer."""
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


def kan_spline_l1(model):
    """L1 penalty on KAN spline coefficients — keeps splines bounded."""
    base = get_base_model(model)
    reg = torch.tensor(0.0, device=next(base.parameters()).device)
    for predictor in base.predictors:
        for layer in predictor.kan.act_fun:
            if hasattr(layer, 'coef'):
                reg = reg + layer.coef.abs().mean()
    return reg


def update_kan_grids(model, y_val, x_val):
    """
    Update KAN spline grids from the actual residual-norm distribution.
    Called once after warmup so splines have better resolution where
    the log1p-norm inputs actually land (which may be far from [-2,4] center).
    Re-initialises theta_bias after grid update to preserve warm-start.
    """
    base = get_base_model(model)
    base.eval()
    with torch.no_grad():
        xh = torch.zeros(y_val.size(0), base._N,
                         device=y_val.device, dtype=y_val.dtype)
        for k in range(base._T):
            res     = y_val - xh @ base.A.T
            features = base.predictors[k].compute_norms(res)  # (batch, in_dim)

            # Update grid and re-interpolate spline coefs
            if hasattr(base.predictors[k].kan, 'update_grid_from_samples'):
                base.predictors[k].kan.update_grid_from_samples(features)

            # Re-zero spline, base, and bias contributions after grid update
            for layer in base.predictors[k].kan.act_fun:
                if hasattr(layer, 'coef'):
                    layer.coef.data.zero_()
                if hasattr(layer, 'scale_base'):
                    layer.scale_base.data.zero_()
            if hasattr(base.predictors[k].kan, 'biases'):
                for b in base.predictors[k].kan.biases:
                    if hasattr(b, 'weight'):
                        b.weight.data.zero_()

            # Re-calibrate theta_bias so output still matches the warm-start value
            kan_residual = base.predictors[k].kan(
                torch.zeros(1, features.size(1), device=y_val.device)).item()
            # Recover the desired theta: c = softplus(theta_bias + old_kan_out)
            c = F.softplus(base.predictors[k].theta_bias).item()
            c = max(c, 1e-6)
            inv_sp = math.log(math.expm1(c))
            base.predictors[k].theta_bias.data.fill_(inv_sp - kan_residual)

            # Advance xh for next layer using current W and theta
            W     = base.Ws[k]
            theta = base.predictors[k](res)
            v     = xh + res @ W.T
            xh    = shrink_free(v, theta)

    base.train()
    print("  KAN grids updated from val-set residual distribution.")


def run_phase(model, A_t, pnz, snr, device, n_epochs, batches_per_epoch,
              batch_size, optimizer, phase_name, y_val, x_val, grad_clip):
    """Generic training loop for Phase A or B."""
    if n_epochs == 0:
        return []

    print(f"\n{'='*55}")
    print(f"  {phase_name}  ({n_epochs} epochs x {batches_per_epoch} batches)")
    print(f"{'='*55}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-8)

    use_amp = (device == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    history    = []
    best_nmse  = float('inf')
    best_state = None
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    base = get_base_model(model)
    model.train()

    for epoch in range(1, n_epochs + 1):
        pool_y, pool_x = _prefetch(A_t, batches_per_epoch, batch_size,
                                   pnz, snr, device)
        epoch_loss = 0.0
        valid_batches = 0
        for b in range(batches_per_epoch):
            y, x = pool_y[b], pool_x[b]
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', enabled=use_amp):
                xhs = model(y)
                loss = sum(nmse_loss(xh, x) for xh in xhs[1:]) / base._T
                loss = loss + REG_LAMBDA * kan_spline_l1(model)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            valid_batches += 1
        if valid_batches == 0:
            print(f"  All batches NaN at epoch {epoch} — stopping.")
            break
        epoch_loss /= valid_batches

        scheduler.step()
        val_nmse = eval_nmse(model, y_val, x_val)
        avg_loss = epoch_loss  # already divided by valid_batches above
        history.append(val_nmse)

        is_best = val_nmse < best_nmse
        if is_best:
            best_nmse  = val_nmse
            best_state = {k: v.clone() for k, v in base.state_dict().items()}
            safe_phase = phase_name.lower().replace(' ', '_').replace('-', '_')
            save_checkpoint(
                checkpoint_path(f'{safe_phase}_best.pth'),
                model_state_dict=best_state,
                A=A_t.detach().cpu().numpy(),
                best_nmse=best_nmse,
                phase_name=phase_name,
                epoch=epoch,
            )

        if epoch % CHECKPOINT_EVERY_EPOCHS == 0:
            safe_phase = phase_name.lower().replace(' ', '_').replace('-', '_')
            save_checkpoint(
                checkpoint_path(f'{safe_phase}_latest.pth'),
                model_state_dict=base.state_dict(),
                A=A_t.detach().cpu().numpy(),
                current_nmse=val_nmse,
                best_nmse=best_nmse,
                phase_name=phase_name,
                epoch=epoch,
            )

        if math.isnan(avg_loss) or math.isnan(val_nmse):
            print(f"  NaN detected at epoch {epoch} — restoring best and stopping.")
            break

        suffix = "  [best]" if is_best else ""
        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"avg_loss={avg_loss:.4f}  val_nmse={val_nmse:.2f} dB{suffix}")

    if best_state is not None:
        base.load_state_dict(best_state)
        print(f"  Restored best model: {best_nmse:.2f} dB")

    return history


def _prefetch(A_t, n_batches, batch_size, pnz, snr, device):
    """Pre-generate n_batches on GPU; returns two lists of tensors."""
    ys, xs = [], []
    for _ in range(n_batches):
        y, x = gen_samples_torch(A_t, batch_size, pnz=pnz, SNR=snr, device=device)
        ys.append(y)
        xs.append(x)
    return ys, xs


def train_variant(variant: str, baseline_model, A, cfg, device, n_gpus: int):
    """Full Phase A + Phase B training for one variant."""
    pnz  = cfg.get('pnz', 0.1)
    snr  = np.inf
    A_t  = torch.from_numpy(A).to(device)
    bs   = BATCH_SIZE * max(1, n_gpus)   # effective batch size

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    y_val, x_val = make_val_set(A_t, n=1000, pnz=pnz, snr=snr, device=device)

    print(f"\n{'#'*60}")
    print(f"  Training LISTA-CP-KAN  variant={variant!r}  bs={bs}")
    print(f"{'#'*60}")

    # ------------------------------------------------------------------
    # Check if final checkpoint already exists — skip training if so.
    # ------------------------------------------------------------------
    final_path = checkpoint_path(
        f'lista_cp_kan{"1" if variant == "1norm" else "3"}_final.pth')
    if os.path.exists(final_path):
        print(f"  Final checkpoint found at {final_path!r}. "
              f"Loading and skipping training.")
        ckpt  = torch.load(final_path, map_location=device, weights_only=False)
        model = LISTACPKan(A, T=cfg['T'], lam=cfg.get('lam', 0.4),
                           variant=variant).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        layer_nmse = ckpt.get('val_nmse_per_layer', [])
        return model, layer_nmse

    model = LISTACPKan(A, T=cfg['T'], lam=cfg.get('lam', 0.4),
                       variant=variant).to(device)
    model.load_from_lista_cp(baseline_model)

    # DataParallel is NOT used for KAN model: pyKAN 0.2.8 stores grid
    # tensors as plain attributes (not registered buffers), so replica
    # creation on GPU 1 would fail with a device mismatch error.
    base = model

    init_nmse = eval_nmse(model, y_val, x_val)
    print(f"  Warm-start NMSE (should ~= baseline): {init_nmse:.2f} dB")

    # ------------------------------------------------------------------
    # Phase A Warmup: KAN only, W frozen, lr=1e-7
    # ------------------------------------------------------------------
    # Check if Phase B checkpoint exists (Phase A already done).
    phase_b_resume_path = checkpoint_path(
        f'phase_b___joint_[{variant}]_best.pth')
    if os.path.exists(phase_b_resume_path):
        print(f"  Phase A checkpoint found. Resuming at Phase B.")
        ckpt = torch.load(phase_b_resume_path, map_location=device,
                          weights_only=False)
        base.load_state_dict(ckpt['model_state_dict'])
        history_a   = []
        history_warmup = []
    else:
        for w in base.Ws:
            w.requires_grad_(False)
        for p in base.predictors.parameters():
            p.requires_grad_(True)

        kan_params = list(base.predictors.parameters())

        opt_warmup = optim.Adam(kan_params, lr=LR_KAN_WARMUP)
        history_warmup = run_phase(
            model, A_t, pnz, snr, device,
            n_epochs=PHASE_A_WARMUP_EPOCHS,
            batches_per_epoch=BATCHES_PER_EPOCH,
            batch_size=bs,
            optimizer=opt_warmup,
            phase_name=f"Phase A Warmup [{variant}]",
            y_val=y_val, x_val=x_val,
            grad_clip=GRAD_CLIP,
        )

        warmup_nmse = eval_nmse(model, y_val, x_val)
        print(f"  After warmup NMSE: {warmup_nmse:.2f} dB  "
              f"(diff from warm-start: {warmup_nmse - init_nmse:+.2f} dB)")

        # Update KAN grids to match actual residual-norm distribution
        update_kan_grids(model, y_val, x_val)
        post_grid_nmse = eval_nmse(model, y_val, x_val)
        print(f"  After grid update NMSE: {post_grid_nmse:.2f} dB")

        history_a = history_warmup  # Phase A main is skipped

    # ------------------------------------------------------------------
    # Phase B: Joint fine-tuning (W + KAN together)
    # ------------------------------------------------------------------
    # Save pre-Phase-B state so we can restore if Phase B goes NaN
    pre_b_state = {k: v.clone() for k, v in base.state_dict().items()}
    pre_b_nmse  = eval_nmse(model, y_val, x_val)

    for w in base.Ws:
        w.requires_grad_(True)

    opt_b = optim.Adam([
        {'params': list(base.Ws.parameters()),         'lr': LR_W},
        {'params': list(base.predictors.parameters()), 'lr': LR_KAN,
         'weight_decay': KAN_WEIGHT_DECAY},
    ])

    history_b = run_phase(
        model, A_t, pnz, snr, device,
        n_epochs=PHASE_B_EPOCHS,
        batches_per_epoch=BATCHES_PER_EPOCH,
        batch_size=bs,
        optimizer=opt_b,
        phase_name=f"Phase B - Joint [{variant}]",
        y_val=y_val, x_val=x_val,
        grad_clip=GRAD_CLIP,
    )

    # If Phase B produced NaN or got worse, restore pre-B state
    post_b_nmse = eval_nmse(model, y_val, x_val)
    if math.isnan(post_b_nmse) or post_b_nmse > pre_b_nmse + 1.0:
        print(f"  Phase B degraded ({post_b_nmse:.2f} vs {pre_b_nmse:.2f}). "
              f"Restoring pre-Phase-B state.")
        base.load_state_dict(pre_b_state)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    layer_nmse = eval_layer_nmse(model, y_val, x_val)
    print(f"\n  Final per-layer NMSE [{variant}]:")
    for k, v in enumerate(layer_nmse):
        print(f"    Layer {k+1:2d}: {v:.2f} dB")

    w_params     = base.get_W_param_count()
    kan_params_n = base.get_kan_param_count()
    print(f"\n  Parameter counts:")
    print(f"    W matrices : {w_params:,}")
    print(f"    KAN params : {kan_params_n:,}")
    print(f"    Total      : {w_params + kan_params_n:,}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    save_path = checkpoint_path(
        f'lista_cp_kan{"1" if variant == "1norm" else "3"}_final.pth')
    save_checkpoint(
        save_path,
        model_state_dict=base.state_dict(),
        A=A,
        config=cfg,
        variant=variant,
        val_nmse_per_layer=layer_nmse,
        history_phaseA=history_a,
        history_phaseB=history_b,
    )
    print(f"\n  Checkpoint saved to {save_path!r}")

    return model, layer_nmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='both',
                        choices=['1norm', '3norm', 'both'],
                        help='Which KAN variant to train.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = torch.cuda.device_count() if device == 'cuda' else 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {device}, GPUs: {n_gpus}")

    baseline_path = checkpoint_path(BASELINE_PATH)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline checkpoint {baseline_path!r} not found. "
            f"Run train_baseline.py first.")

    baseline_model, A, cfg = load_baseline(baseline_path, device)

    variants = ['1norm', '3norm'] if args.variant == 'both' else [args.variant]

    results = {}
    for v in variants:
        t0 = time.time()
        model, layer_nmse = train_variant(v, baseline_model, A, cfg, device, n_gpus)
        print(f"\n  Variant {v!r} done in {(time.time()-t0)/60:.1f} min")
        results[v] = layer_nmse

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for v, nmse in results.items():
        final = f"{nmse[-1]:.2f} dB" if nmse else "N/A (no layer NMSE saved)"
        print(f"  {v:6s}: final NMSE (layer 16) = {final}")


if __name__ == '__main__':
    main()
