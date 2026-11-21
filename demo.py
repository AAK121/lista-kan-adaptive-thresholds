"""
demo.py  —  Viva demonstration script

Models shown:
  1. LISTA-CP Baseline          (lista_cp_layer16.pth  or  lista_cp_baseline.pth)
  2. LISTA-CP-KAN Scratch       (lista_cp_kan_scratch_layer07.pth  — partial, 7/16 layers)
  3. LISTA-CP-KAN Fine-tuned    (lista_cp_kan3_final.pth  or  lista_cp_kan1_final.pth)

Input data:
  - The sensing matrix A (250×500) is stored INSIDE the checkpoint file.
    No external matrix file is needed.
  - A test sparse signal x is generated RANDOMLY on the fly using numpy seed 99.
    x is a 500-dim vector with ~10% non-zero Gaussian entries.
  - Measurement y = A·x  is computed directly — no image or dataset file needed.
  - Optionally: if LISTA-CPSS/data/xtest_n500_p10.npy exists, the summary uses
    that standard test set for NMSE. Otherwise the script uses generated data.

Usage (Windows local):
    venv/Scripts/python.exe demo.py

Place these files in D:\\cs754_research_project\\ before running:
    lista_cp_layer16.pth                      ← baseline (you already have this)
    lista_cp_kan_scratch_layer07.pth          ← KAN scratch up to layer 7
    lista_cp_kan3_final.pth  (or kan1)        ← KAN fine-tuned
"""

import os, sys
import numpy as np
import torch
import matplotlib

# Auto-select backend: interactive on local Windows, file-only on Colab/Kaggle
_headless = any(os.path.isdir(p) for p in ['/kaggle', '/content'])
matplotlib.use('Agg' if _headless else 'TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from problem_setup import random_A, gen_samples_torch, compute_nmse_db_torch
from lista_cp_torch import LISTACPTorch, shrink_free

# ── Where to find checkpoints ────────────────────────────────────────────────
def _find_dir():
    for d in ['/content/drive/MyDrive/cs754_kan_checkpoints',
              '/kaggle/working', '.']:
        if os.path.isdir(d):
            return d
    return '.'

CKPT_DIR = os.environ.get('CHECKPOINT_DIR', _find_dir())
SAVE_DIR = 'demo_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

M, N, T = 250, 500, 16
SEED    = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device        : {device}")
print(f"Checkpoint dir: {CKPT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_file(*names):
    """Return first existing path from names, checking CKPT_DIR then '.'."""
    for name in names:
        for base in [CKPT_DIR, '.']:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    return None


def clean_state(state_dict):
    """Strip _orig_mod. prefix added by torch.compile."""
    if any(k.startswith('_orig_mod.') for k in state_dict):
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    return state_dict


def get_layer_nmse(model, y, x, max_layers=None):
    model.eval()
    with torch.no_grad():
        xhs = model(y)
    limit = (max_layers or len(xhs) - 1)
    nmse  = [compute_nmse_db_torch(xhs[k], x).item() for k in range(1, limit + 1)]
    return xhs, nmse


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load models
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  LOADING MODELS")
print("=" * 60)

# ── 1a. Baseline ─────────────────────────────────────────────────────────────
path = find_file('lista_cp_baseline.pth', 'lista_cp_layer16.pth')
if path is None:
    sys.exit("ERROR: lista_cp_layer16.pth not found. "
             f"Place it in {os.path.abspath('.')}")

ckpt     = torch.load(path, map_location=device, weights_only=False)
A        = ckpt['A']                         # sensing matrix stored inside checkpoint
cfg      = ckpt['config']
baseline = LISTACPTorch(A, T=cfg['T'], lam=cfg.get('lam', 0.4)).to(device)
baseline.load_state_dict(clean_state(ckpt['model_state_dict']))
baseline.eval()
print(f"[1] Baseline loaded        : {path}")
print(f"    Stored A shape         : {A.shape}  (M=250, N=500)")
print(f"    Config                 : {cfg}")

A_t = torch.from_numpy(A).to(device)

# ── 1b. KAN Scratch (partial — up to layer 7) ────────────────────────────────
kan_scratch = None
kan_scratch_layers = 0
try:
    from lista_cp_kan import LISTACPKan

    # Try highest available layer first
    scratch_path = None
    for lyr in range(T, 0, -1):
        p = find_file(f'lista_cp_kan_scratch_layer{lyr:02d}.pth',
                      f'lista_cp_kan3_scratch.pth',
                      f'lista_cp_kan1_scratch.pth')
        if p:
            scratch_path = p
            break

    if scratch_path:
        ckpt_s  = torch.load(scratch_path, map_location=device, weights_only=False)
        variant = ckpt_s.get('variant', '3norm')
        cfg_s   = ckpt_s.get('config', cfg)
        kan_scratch = LISTACPKan(A, T=cfg_s['T'], lam=cfg_s.get('lam', 0.4),
                                 variant=variant).to(device)
        kan_scratch.load_state_dict(clean_state(ckpt_s['model_state_dict']))
        kan_scratch.eval()
        kan_scratch_layers = ckpt_s.get('layer_completed', T)
        print(f"\n[2] KAN Scratch loaded     : {scratch_path}")
        print(f"    Variant                : {variant}")
        print(f"    Layers trained         : {kan_scratch_layers}/{T}")
        print(f"    (Only layers 1–{kan_scratch_layers} are meaningful for evaluation)")
    else:
        print("\n[2] KAN Scratch            : NOT FOUND (skipping)")
except ImportError:
    print("\n[2] KAN Scratch            : pykan not installed (skipping)")

# ── 1c. KAN Fine-tuned ────────────────────────────────────────────────────────
kan_ft = None
kan_ft_variant = None
try:
    from lista_cp_kan import LISTACPKan
    ft_path = find_file('lista_cp_kan3_final.pth', 'lista_cp_kan1_final.pth')
    if ft_path:
        ckpt_f  = torch.load(ft_path, map_location=device, weights_only=False)
        kan_ft_variant = ckpt_f.get('variant', '3norm')
        cfg_f   = ckpt_f.get('config', cfg)
        kan_ft  = LISTACPKan(A, T=cfg_f['T'], lam=cfg_f.get('lam', 0.4),
                              variant=kan_ft_variant).to(device)
        kan_ft.load_state_dict(clean_state(ckpt_f['model_state_dict']))
        kan_ft.eval()
        print(f"\n[3] KAN Fine-tuned loaded  : {ft_path}")
        print(f"    Variant                : {kan_ft_variant}")
        print(f"    All {T} layers present")
    else:
        print("\n[3] KAN Fine-tuned         : NOT FOUND (skipping)")
except ImportError:
    print("\n[3] KAN Fine-tuned         : pykan not installed (skipping)")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Generate test sparse signal
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  INPUT DATA")
print("=" * 60)

np.random.seed(99)
torch.manual_seed(99)
y_demo, x_demo = gen_samples_torch(A_t, batch_size=1, pnz=0.1,
                                   SNR=np.inf, device=device)
x_np = x_demo[0].cpu().numpy()
y_np = y_demo[0].cpu().numpy()
nnz  = int((x_np != 0).sum())

print(f"Signal x      : ℝ^{N},  {nnz} non-zeros ({nnz/N:.1%} sparsity)")
print(f"              : entries ~ N(0,1) at support positions, 0 elsewhere")
print(f"Measurement y : ℝ^{M}  computed as  y = A·x  (noiseless)")
print(f"              : A is fixed (seed=42), loaded from checkpoint")
print(f"Goal          : recover x from y using {T}-layer unrolled network")
print(f"              : NO external image or dataset file needed")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Run forward pass — layer by layer
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  LAYER-BY-LAYER NMSE")
print("=" * 60)

xhs_base, nmse_base = get_layer_nmse(baseline, y_demo, x_demo)

print(f"\n{'Layer':>6}  {'Baseline':>10}", end='')
if kan_scratch:
    print(f"  {'KAN-Scratch':>12}  {'Δ scratch':>10}", end='')
if kan_ft:
    print(f"  {'KAN-FineTune':>13}  {'Δ ft':>8}", end='')
print()
print("-" * (38 + (26 if kan_scratch else 0) + (24 if kan_ft else 0)))

xhs_scratch, nmse_scratch = (get_layer_nmse(kan_scratch, y_demo, x_demo,
                              max_layers=kan_scratch_layers)
                              if kan_scratch else (None, []))
xhs_ft, nmse_ft = (get_layer_nmse(kan_ft, y_demo, x_demo)
                   if kan_ft else (None, []))

for k in range(T):
    line = f"  {k+1:>4d}  {nmse_base[k]:>10.2f}"
    if kan_scratch:
        if k < len(nmse_scratch):
            d = nmse_scratch[k] - nmse_base[k]
            line += f"  {nmse_scratch[k]:>12.2f}  {d:>+10.2f}"
        else:
            line += f"  {'(not trained)':>12}  {'—':>10}"
    if kan_ft:
        if k < len(nmse_ft):
            d = nmse_ft[k] - nmse_base[k]
            line += f"  {nmse_ft[k]:>13.2f}  {d:>+8.2f}"
    print(line)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  GENERATING PLOTS")
print("=" * 60)

# ── Plot 1: Problem setup ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

ax = fig.add_subplot(gs[0])
ax.imshow(A, aspect='auto', cmap='RdBu', interpolation='none')
ax.set_title(f'Sensing Matrix A  ({M}×{N})', fontsize=10)
ax.set_xlabel('N = 500'); ax.set_ylabel('M = 250')

ax = fig.add_subplot(gs[1])
ax.plot(x_np, 'C0', lw=0.7, label='True signal x')
ax.set_title(f'Sparse signal x  (nnz={nnz}/{N})', fontsize=10)
ax.set_xlabel('Index n'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2])
ax.plot(y_np, 'C1', lw=0.7)
ax.set_title(f'Measurements y = Ax  (dim {M})', fontsize=10)
ax.set_xlabel('Index m'); ax.grid(True, alpha=0.3)

fig.suptitle('Compressed Sensing Problem Setup  —  y = Ax,  recover sparse x',
             fontsize=11, fontweight='bold')
fig.tight_layout()
p = os.path.join(SAVE_DIR, 'plot1_problem_setup.png')
fig.savefig(p, dpi=150, bbox_inches='tight')
print(f"  Saved: {p}")
plt.show()

# ── Plot 2: Reconstruction at different layers ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('LISTA-CP Reconstruction Quality by Layer', fontsize=11, fontweight='bold')

for ax, (layer, label) in zip(axes, [
        (1,  'Layer 1  (initial step)'),
        (T//2, f'Layer {T//2}  (halfway)'),
        (T,  f'Layer {T}  (final)')]):
    xh_np = xhs_base[layer][0].cpu().numpy()
    ax.plot(x_np,  'C0', lw=0.8, alpha=0.6, label='True x')
    ax.plot(xh_np, 'C2', lw=0.8, alpha=0.9,
            label=f'Estimated x̂  ({nmse_base[layer-1]:.1f} dB)')
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Index n'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

fig.tight_layout()
p = os.path.join(SAVE_DIR, 'plot2_reconstruction_layers.png')
fig.savefig(p, dpi=150, bbox_inches='tight')
print(f"  Saved: {p}")
plt.show()

# ── Plot 3: NMSE convergence — all models ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(range(1, T+1), nmse_base, 'C0-o', lw=2.0, ms=5,
        label='LISTA-CP Baseline (16 layers)')

if kan_scratch and nmse_scratch:
    ax.plot(range(1, len(nmse_scratch)+1), nmse_scratch, 'C2-s', lw=2.0, ms=5,
            label=f'LISTA-CP-KAN Scratch ({kan_scratch_layers}/{T} layers trained)',
            linestyle='--')
    ax.axvline(kan_scratch_layers, color='C2', lw=0.8, ls=':', alpha=0.5)

if kan_ft and nmse_ft:
    ax.plot(range(1, T+1), nmse_ft, 'C3-^', lw=2.0, ms=5,
            label=f'LISTA-CP-KAN Fine-tuned ({kan_ft_variant})')

ax.axhline(-30.0, color='gray', ls='--', lw=1.0, label='-30 dB target')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('NMSE (dB)', fontsize=12)
ax.set_title('NMSE per Layer — All Models Compared', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.35)
ax.set_xticks(range(1, T+1))
fig.tight_layout()
p = os.path.join(SAVE_DIR, 'plot3_nmse_all_models.png')
fig.savefig(p, dpi=150, bbox_inches='tight')
print(f"  Saved: {p}")
plt.show()

# ── Plot 4: Adaptive threshold comparison (KAN vs fixed) ─────────────────────
kan_to_plot = kan_scratch or kan_ft
kan_label   = ('KAN Scratch' if kan_scratch else 'KAN Fine-tuned')

if kan_to_plot:
    kan_layers_plot = kan_scratch_layers if kan_scratch else T
    fig, ax = plt.subplots(figsize=(9, 4))

    # Collect KAN adaptive thresholds layer by layer
    thresholds = []
    kan_to_plot.eval()
    with torch.no_grad():
        xh = torch.zeros(1, N, device=device)
        for k in range(kan_layers_plot):
            W   = kan_to_plot.Ws[k]
            res = y_demo - xh @ kan_to_plot.A.T
            th  = kan_to_plot.predictors[k](res)   # (1, 1)
            thresholds.append(th.item())
            v   = xh + res @ W.T
            xh  = shrink_free(v, th)

    base_thetas = [baseline.thetas[k].item() for k in range(T)]

    ax.plot(range(1, T+1), base_thetas, 'C0-o', lw=1.8, ms=5,
            label='Baseline: fixed scalar θ_k  (same for every sample)')
    ax.plot(range(1, len(thresholds)+1), thresholds, 'C2-s', lw=1.8, ms=5,
            linestyle='--',
            label=f'{kan_label}: adaptive θ_k  (changes per sample, per layer)')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Threshold θ', fontsize=12)
    ax.set_title('Key Difference: Fixed vs Adaptive Threshold per Layer',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p = os.path.join(SAVE_DIR, 'plot4_adaptive_threshold.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Problem : recover x ∈ ℝ^{N} (nnz={nnz}) from y ∈ ℝ^{M}")
print(f"  Input   : randomly generated sparse signal (seed=99)")
print(f"            A matrix loaded from checkpoint (seed=42, fixed)")

print(f"\n  Model                     Params      Layer 16 NMSE")
print(f"  " + "-"*50)

bp = sum(p.numel() for p in baseline.parameters())
print(f"  Baseline LISTA-CP         {bp:>8,}    {nmse_base[-1]:.2f} dB")

if kan_scratch:
    ksp = sum(p.numel() for p in kan_scratch.parameters())
    best = nmse_scratch[-1] if nmse_scratch else float('nan')
    print(f"  KAN Scratch (layer {kan_scratch_layers}/{T})    "
          f"{ksp:>8,}    {best:.2f} dB  (at layer {kan_scratch_layers})")

if kan_ft:
    kfp = sum(p.numel() for p in kan_ft.parameters())
    print(f"  KAN Fine-tuned            {kfp:>8,}    {nmse_ft[-1]:.2f} dB")

print(f"\n  Plots saved to: {os.path.abspath(SAVE_DIR)}/")
print("=" * 60)
