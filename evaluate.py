"""
evaluate.py
Full evaluation of LISTA-CP baseline and both KAN variants.

Evaluations:
  1. NMSE-per-layer curves for: noiseless, SNR=40, 30, 20 dB
  2. Final NMSE table (layer 16)
  3. Layers needed to first reach -30 dB NMSE (noiseless)
  4. Parameter count comparison
  5. KAN adaptivity checks:
       - All predicted thetas > 0
       - Std of theta across samples per layer (should be > 0 if adapting)
  6. Spline visualisation for layer 8

Usage:
  python evaluate.py
"""

import os, math, tempfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from problem_setup import load_test_data, compute_nmse_db_torch
from lista_cp_torch import LISTACPTorch
from lista_cp_kan import LISTACPKan

BASELINE_PATH   = 'lista_cp_baseline.pth' if os.path.exists('lista_cp_baseline.pth') else 'lista_cp_layer16.pth'
KAN1_PATH       = 'lista_cp_kan1_final.pth'
KAN3_PATH       = 'lista_cp_kan3_final.pth'
XTEST_PATH      = 'LISTA-CPSS/data/xtest_n500_p10.npy'
RESULTS_DIR     = 'results'

SNR_CONDITIONS  = [np.inf, 40.0, 30.0, 20.0]
SNR_LABELS      = ['noiseless', 'SNR=40dB', 'SNR=30dB', 'SNR=20dB']

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------

def load_baseline_model(path, device='cpu'):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    A     = ckpt['A']
    cfg   = ckpt['config']
    model = LISTACPTorch(A, T=cfg['T'], lam=cfg.get('lam', 0.4)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, A, cfg


def load_kan_model(path, device='cpu'):
    ckpt    = torch.load(path, map_location=device, weights_only=False)
    A       = ckpt['A']
    cfg     = ckpt['config']
    variant = ckpt['variant']
    model   = LISTACPKan(A, T=cfg['T'], lam=cfg.get('lam', 0.4),
                         variant=variant).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, A, cfg, variant


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def eval_layer_nmse_np(model, A, snr, xtest_path, device='cpu'):
    """
    Returns list of NMSE (dB) at each of T layers.
    Loads test data afresh for the requested SNR.
    """
    y_test, x_test = load_test_data(xtest_path, A, SNR=snr)
    y_t = torch.from_numpy(y_test).to(device)
    x_t = torch.from_numpy(x_test).to(device)

    with torch.no_grad():
        xhs = model(y_t)
        layer_nmse = [compute_nmse_db_torch(xh, x_t).item() for xh in xhs[1:]]
    return layer_nmse


def layers_to_reach(layer_nmse, threshold=-30.0):
    """Return the 1-based layer index where NMSE first drops below threshold."""
    for k, v in enumerate(layer_nmse):
        if v <= threshold:
            return k + 1
    return None  # never reached


# -----------------------------------------------------------------------
# KAN adaptivity checks
# -----------------------------------------------------------------------

def check_adaptivity(model, A, snr, xtest_path, device='cpu', variant='1norm'):
    """
    For a LISTACPKan model, compute:
      - min predicted theta across all samples and layers (must be > 0)
      - std of theta per layer across test samples
    """
    from problem_setup import gen_samples_torch
    y_test, x_test = load_test_data(xtest_path, A, SNR=snr)
    y_t = torch.from_numpy(y_test).to(device)

    model.eval()
    all_thetas = []   # list of T tensors, each (n_test,)

    # Run forward manually to collect per-layer thetas
    batch = y_t.size(0)
    xh = torch.zeros(batch, model._N, device=device)
    with torch.no_grad():
        for k in range(model._T):
            W = model.Ws[k]
            res = y_t - xh @ model.A.T
            theta = model.predictors[k](res)  # (batch, 1)
            all_thetas.append(theta.squeeze(1))  # (batch,)
            v  = xh + res @ W.T
            from lista_cp_torch import shrink_free
            xh = shrink_free(v, theta)

    all_thetas = torch.stack(all_thetas, dim=0)  # (T, batch)

    min_theta = all_thetas.min().item()
    layer_std = all_thetas.std(dim=1).tolist()  # (T,) std over samples
    layer_mean = all_thetas.mean(dim=1).tolist()

    return min_theta, layer_std, layer_mean, all_thetas


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

def plot_layer_nmse(all_results, snr_label, save_path):
    """
    all_results: dict { model_name: list_of_T_nmse_values }
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {'Baseline': 'o', 'KAN-1norm': 's', 'KAN-3norm': '^'}
    colors  = {'Baseline': '#2196F3', 'KAN-1norm': '#E91E63', 'KAN-3norm': '#4CAF50'}

    for name, nmse in all_results.items():
        layers = list(range(1, len(nmse) + 1))
        ax.plot(layers, nmse,
                marker=markers.get(name, 'x'),
                color=colors.get(name, 'k'),
                label=name, linewidth=1.8, markersize=5)

    ax.axhline(-30.0, color='gray', linestyle='--', linewidth=0.8, label='-30 dB')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title(f'LISTA-CP vs KAN variants — {snr_label}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.set_xticks(range(1, len(next(iter(all_results.values()))) + 1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_theta_std(layer_std_dict, save_path):
    """Plot per-layer theta std for KAN variants (adaptivity visualisation)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, stds in layer_std_dict.items():
        ax.plot(range(1, len(stds) + 1), stds, marker='o', label=name,
                linewidth=1.8, markersize=5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Std(theta) across test samples', fontsize=12)
    ax.set_title('KAN threshold adaptivity — std per layer', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_kan_spline_layer8(kan_model, variant, save_path):
    """
    Visualise the learned KAN spline function at layer 8 (index 7).
    For '1norm': a 1D curve (L2 norm → theta).
    For '3norm': three 1D splines (one per input feature).
    """
    predictor = kan_model.predictors[7]  # layer 8 (0-indexed)
    predictor.eval()

    if variant == '1norm':
        # Sweep L2 norm from 0 to 3 (normalised units)
        x_vals = torch.linspace(0.0, 3.0, 200).unsqueeze(1)  # (200, 1)
        with torch.no_grad():
            thetas = predictor(
                x_vals * math.sqrt(predictor.M)  # undo normalisation for residual
            ).squeeze(1).numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_vals.squeeze().numpy(), thetas, linewidth=2, color='#E91E63')
        ax.set_xlabel('Normalised L2 residual norm', fontsize=11)
        ax.set_ylabel('theta (threshold)', fontsize=11)
        ax.set_title('Layer 8 KAN spline — 1norm variant', fontsize=12)
        ax.grid(True, alpha=0.35)
        fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

    else:  # 3norm
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        names = ['L1/m', 'L2/sqrt(m)', 'L-inf']
        M_ = predictor.M

        for feat_idx, (ax, feat_name) in enumerate(zip(axes, names)):
            # Sweep one feature, hold others at typical value ~1.0
            sweep = torch.linspace(0.0, 3.0, 200)
            feat_tensor = torch.ones(200, 3) * 1.0
            feat_tensor[:, feat_idx] = sweep

            # Convert features back to a residual with matching norms
            # (simplest: create a residual tensor that gives the desired norms)
            # Instead, directly pass feature vectors through compute_norms path
            with torch.no_grad():
                raw = predictor.kan(feat_tensor)
                raw = raw + predictor.theta_bias
                theta = torch.nn.functional.softplus(raw).squeeze(1).numpy()

            ax.plot(sweep.numpy(), theta, linewidth=2)
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel('theta' if feat_idx == 0 else '', fontsize=10)
            ax.set_title(f'Feature {feat_idx+1}: {feat_name}', fontsize=10)
            ax.grid(True, alpha=0.35)

        fig.suptitle('Layer 8 KAN splines — 3norm variant', fontsize=13)
        fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

    print(f"  Saved {save_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------
    # Load all available models
    # ------------------------------------------------------------------
    models  = {}
    A_ref   = None
    cfg_ref = None

    if os.path.exists(BASELINE_PATH):
        bm, A_ref, cfg_ref = load_baseline_model(BASELINE_PATH, device)
        models['Baseline'] = (bm, A_ref)
        print(f"Loaded baseline: {BASELINE_PATH}")
    else:
        print(f"WARNING: {BASELINE_PATH} not found; skipping baseline.")

    for label, path, variant in [('KAN-1norm', KAN1_PATH, '1norm'),
                                  ('KAN-3norm', KAN3_PATH, '3norm')]:
        if os.path.exists(path):
            km, A_k, cfg_k, var = load_kan_model(path, device)
            if A_ref is None:
                A_ref, cfg_ref = A_k, cfg_k
            models[label] = (km, A_k)
            print(f"Loaded {label}: {path}")
        else:
            print(f"WARNING: {path} not found; skipping {label}.")

    if not models:
        raise RuntimeError("No model checkpoints found. Train models first.")

    if not os.path.exists(XTEST_PATH):
        print(f"WARNING: Test file {XTEST_PATH!r} not found. "
              f"Generating synthetic test data.")
        # Fall back: generate test data
        from problem_setup import gen_samples_torch
        A_t  = torch.from_numpy(A_ref)
        y, x = gen_samples_torch(A_t, 1000, pnz=0.1, SNR=np.inf)
        tmp_path = os.path.join(tempfile.gettempdir(), '_xtest_tmp.npy')
        np.save(tmp_path, x.numpy().T)  # save column-major
        xtest_path_eff = tmp_path
    else:
        xtest_path_eff = XTEST_PATH

    # ------------------------------------------------------------------
    # 1. NMSE-per-layer curves for each SNR condition
    # ------------------------------------------------------------------
    print("\n[1] Computing NMSE per layer ...")
    nmse_table = {}   # nmse_table[model_name][snr_label] = list_of_T_values

    for name, (model, A_m) in models.items():
        nmse_table[name] = {}
        for snr, slabel in zip(SNR_CONDITIONS, SNR_LABELS):
            nmse_list = eval_layer_nmse_np(model, A_m, snr,
                                           xtest_path_eff, device)
            nmse_table[name][slabel] = nmse_list
            print(f"  {name:12s}  {slabel:12s}  "
                  f"final={nmse_list[-1]:.2f} dB")

    # ------------------------------------------------------------------
    # 2. Per-condition plots
    # ------------------------------------------------------------------
    print("\n[2] Generating NMSE-per-layer plots ...")
    for slabel in SNR_LABELS:
        all_res = {name: nmse_table[name][slabel]
                   for name in models if slabel in nmse_table[name]}
        plot_layer_nmse(all_res, slabel,
                        os.path.join(RESULTS_DIR, f'nmse_layer_{slabel.replace("=","").replace(" ","_")}.png'))

    # ------------------------------------------------------------------
    # 3. Summary table
    # ------------------------------------------------------------------
    print("\n[3] Summary Table")
    print("-" * 70)
    header = f"{'Model':12s} | " + " | ".join(f"{s:12s}" for s in SNR_LABELS)
    print(header)
    print("-" * 70)
    for name in models:
        row_vals = [f"{nmse_table[name][s][-1]:12.2f}"
                    for s in SNR_LABELS if s in nmse_table[name]]
        print(f"{name:12s} | " + " | ".join(row_vals))

    # ------------------------------------------------------------------
    # 4. Layers to reach -30 dB (noiseless)
    # ------------------------------------------------------------------
    print("\n[4] Layers to first reach -30 dB NMSE (noiseless)")
    for name in models:
        nl = nmse_table[name].get('noiseless')
        if nl:
            k = layers_to_reach(nl, -30.0)
            print(f"  {name:12s}: layer {k if k else 'never'}")

    # ------------------------------------------------------------------
    # 5. Parameter counts
    # ------------------------------------------------------------------
    print("\n[5] Parameter counts")
    for name, (model, _) in models.items():
        total = sum(p.numel() for p in model.parameters())
        print(f"  {name:12s}: {total:>10,} params")
        if hasattr(model, 'get_kan_param_count'):
            print(f"    W params : {model.get_W_param_count():>10,}")
            print(f"    KAN params: {model.get_kan_param_count():>10,}")

    # ------------------------------------------------------------------
    # 6. Adaptivity checks for KAN models
    # ------------------------------------------------------------------
    print("\n[6] KAN adaptivity checks")
    theta_std_dict = {}
    for name, (model, A_m) in models.items():
        if not isinstance(model, LISTACPKan):
            continue
        min_t, stds, means, _ = check_adaptivity(
            model, A_m, np.inf, xtest_path_eff, device,
            variant=model.variant)

        print(f"\n  {name}:")
        print(f"    Min theta across all samples/layers: {min_t:.6f}  "
              f"({'OK' if min_t > 0 else 'FAIL — negative theta!'})")
        near_zero = sum(1 for s in stds if s < 1e-4)
        print(f"    Layers with std(theta) < 1e-4: {near_zero}/{len(stds)}  "
              f"({'WARN: low adaptivity' if near_zero > 8 else 'OK'})")
        print(f"    Per-layer std: "
              f"{[round(s, 5) for s in stds]}")
        theta_std_dict[name] = stds

    if theta_std_dict:
        plot_theta_std(theta_std_dict,
                       os.path.join(RESULTS_DIR, 'theta_std_per_layer.png'))

    # ------------------------------------------------------------------
    # 7. Spline visualisation for layer 8
    # ------------------------------------------------------------------
    print("\n[7] Spline visualisation (layer 8)")
    for name, (model, _) in models.items():
        if not isinstance(model, LISTACPKan):
            continue
        variant = model.variant
        suf     = '1' if variant == '1norm' else '3'
        save_sp = os.path.join(RESULTS_DIR, f'spline_layer8_kan{suf}.png')
        plot_kan_spline_layer8(model, variant, save_sp)

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
