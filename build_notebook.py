"""Run this once to generate demo.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))

# ─── Cell 1: Title ────────────────────────────────────────────────────────────
md("""# LISTA-CP vs Adaptive Thresholding — Demo
### CS 754: Advanced Image Processing

**Models compared:**
| # | Model | File |
|---|-------|------|
| 1 | LISTA-CP Baseline (16 layers) | `lista_cp_layer16.pth` |
| 2 | LISTA-CP-KAN Scratch (7/16 layers) | `lista_cp_kan_scratch_layer07.pth` |
| 3 | LISTA-CP-KAN Fine-tuned | `phase_b___joint_[3norm]_latest.pth` |

**Input:** Randomly generated sparse signal — no external dataset file needed.
The sensing matrix `A` is stored inside each checkpoint file.
""")

# ─── Cell 2: Imports ──────────────────────────────────────────────────────────
md("## 1. Imports & Setup")
code("""
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
plt.rcParams['figure.dpi'] = 120

# ── Change this if your project is elsewhere ─────────────────────────────────
PROJECT_DIR = r'D:/cs754_research_project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# ── Fast imports (no KAN yet) ─────────────────────────────────────────────────
from problem_setup import gen_samples_torch, compute_nmse_db_torch
from lista_cp_torch import LISTACPTorch, shrink_free

device = 'cuda' if torch.cuda.is_available() else 'cpu'
M, N, T = 250, 500, 16
print(f"Device          : {device}")
print(f"Working dir     : {os.getcwd()}")
print("Basic imports done. Run next cell to load models (KAN import is slow ~30-60s).")
""")

# ─── Cell 3: Load models ──────────────────────────────────────────────────────
md("## 2. Load Checkpoints\n\n> ⏳ **This cell takes ~30–60 seconds** — pykan builds 16×3 spline networks on first load. The `[*]` indicator means it is still running, not frozen.")
code("""
import time

def clean_state(sd):
    if any(k.startswith('_orig_mod.') for k in sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    return sd

def load_raw(path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    if 'config' in c: print(f"  config : {c['config']}")
    if 'variant' in c: print(f"  variant: {c['variant']}")
    if 'layer_completed' in c: print(f"  layers completed: {c['layer_completed']}")
    return c

# ── 1. Baseline  (fast — no KAN) ─────────────────────────────────────────────
print("=== [1/3] Baseline ===")
t0 = time.time()
ckpt_base = load_raw('lista_cp_layer16.pth')
A   = ckpt_base['A']
cfg = ckpt_base['config']
baseline = LISTACPTorch(A, T=cfg['T'], lam=cfg.get('lam', 0.4)).to(device)
baseline.load_state_dict(clean_state(ckpt_base['model_state_dict']))
baseline.eval()
A_t = torch.from_numpy(A).to(device)
print(f"  Done in {time.time()-t0:.1f}s  |  "
      f"params: {sum(p.numel() for p in baseline.parameters()):,}")

# ── 2. KAN Scratch — importing pykan is slow, shown once ─────────────────────
print("\\n=== [2/3] KAN Scratch  (importing pykan... this takes ~30-60s) ===")
t0 = time.time()
from lista_cp_kan import LISTACPKan   # slow import — only here, not cell 1

ckpt_scratch = load_raw('lista_cp_kan_scratch_layer07.pth')
variant_s    = ckpt_scratch.get('variant', '3norm')
cfg_s        = ckpt_scratch.get('config', cfg)
kan_scratch  = LISTACPKan(A, T=cfg_s['T'], lam=cfg_s.get('lam', 0.4),
                           variant=variant_s).to(device)
kan_scratch.load_state_dict(clean_state(ckpt_scratch['model_state_dict']))
kan_scratch.eval()
KAN_SCRATCH_LAYERS = ckpt_scratch.get('layer_completed', 7)
print(f"  Done in {time.time()-t0:.1f}s  |  variant={variant_s}, "
      f"layers trained={KAN_SCRATCH_LAYERS}/{T}  |  "
      f"params: {sum(p.numel() for p in kan_scratch.parameters()):,}")

# ── 3. KAN Fine-tuned ─────────────────────────────────────────────────────────
print("\\n=== [3/3] KAN Fine-tuned ===")
t0 = time.time()
ckpt_ft   = load_raw('phase_b___joint_[3norm]_latest.pth')
variant_f = ckpt_ft.get('variant', '3norm')
cfg_f     = ckpt_ft.get('config', cfg)
kan_ft    = LISTACPKan(A, T=cfg_f['T'], lam=cfg_f.get('lam', 0.4),
                       variant=variant_f).to(device)
kan_ft.load_state_dict(clean_state(ckpt_ft['model_state_dict']))
kan_ft.eval()
print(f"  Done in {time.time()-t0:.1f}s  |  variant={variant_f}  |  "
      f"params: {sum(p.numel() for p in kan_ft.parameters()):,}")

print("\\nAll models loaded.")
""")

# ─── Cell 4: Generate input ───────────────────────────────────────────────────
md("""## 3. Generate Test Input

**What is the input?**
- `A` (250×500) — the sensing/measurement matrix, loaded from the checkpoint (same seed=42 across all models)
- `x` — a 500-dimensional sparse signal: ~10% entries are random Gaussian, rest are exactly 0
- `y = A·x` — the compressed measurement vector (250-dimensional), computed directly

The goal of each model is to recover `x` from only `y` (and `A`).
""")
code("""
np.random.seed(99)
torch.manual_seed(99)

y_demo, x_demo = gen_samples_torch(A_t, batch_size=1, pnz=0.1, SNR=np.inf, device=device)
x_np = x_demo[0].cpu().numpy()
y_np = y_demo[0].cpu().numpy()
nnz  = int((x_np != 0).sum())

print(f"Signal x   : dimension {N},  non-zeros = {nnz}  ({nnz/N:.1%} sparsity)")
print(f"             non-zero values ~ N(0,1),  support chosen randomly")
print(f"Measurement: dimension {M}  →  y = A·x  (noiseless)")
print(f"Task       : recover x from y using a {T}-layer unrolled network")
print(f"             NMSE measures how close x̂ is to true x  (lower = better)")
""")

# ─── Cell 5: Problem setup plot ───────────────────────────────────────────────
md("## 4. Problem Setup — Visualised")
code("""
fig = plt.figure(figsize=(13, 3.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

ax = fig.add_subplot(gs[0])
ax.imshow(A, aspect='auto', cmap='RdBu', interpolation='none')
ax.set_title(f'Sensing Matrix A  ({M}×{N})', fontsize=10)
ax.set_xlabel('N = 500'); ax.set_ylabel('M = 250')

ax = fig.add_subplot(gs[1])
ax.plot(x_np, 'C0', lw=0.7)
ax.set_title(f'Sparse Signal x   (nnz = {nnz} / {N})', fontsize=10)
ax.set_xlabel('Index n'); ax.set_ylabel('Amplitude'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[2])
ax.plot(y_np, 'C1', lw=0.7)
ax.set_title(f'Measurements y = Ax   (dim {M})', fontsize=10)
ax.set_xlabel('Index m'); ax.set_ylabel('Amplitude'); ax.grid(True, alpha=0.3)

fig.suptitle('Compressed Sensing:  y = Ax,  recover sparse x  (M=250 << N=500)',
             fontsize=11, fontweight='bold')
plt.tight_layout(); plt.show()
""")

# ─── Cell 6: Layer-by-layer output ───────────────────────────────────────────
md("## 5. Run All Models — Layer-by-Layer NMSE")
code("""
def layer_nmse(model, y, x, max_layers=None):
    model.eval()
    with torch.no_grad():
        xhs = model(y)
    lim  = max_layers or (len(xhs) - 1)
    nmse = [compute_nmse_db_torch(xhs[k], x).item() for k in range(1, lim + 1)]
    return xhs, nmse

xhs_base,    nmse_base    = layer_nmse(baseline,    y_demo, x_demo)
xhs_scratch, nmse_scratch = layer_nmse(kan_scratch, y_demo, x_demo, KAN_SCRATCH_LAYERS)
xhs_ft,      nmse_ft      = layer_nmse(kan_ft,      y_demo, x_demo)

print(f"{'Layer':>5}  {'Baseline':>10}  {'KAN Scratch':>13}  {'KAN Fine-tune':>14}")
print("-" * 50)
for k in range(T):
    s = f"{nmse_scratch[k]:>13.2f}" if k < len(nmse_scratch) else f"{'(not trained)':>13}"
    print(f"  {k+1:>3}  {nmse_base[k]:>10.2f}  {s}  {nmse_ft[k]:>14.2f}")
""")

# ─── Cell 7: Reconstruction plot ─────────────────────────────────────────────
md("## 6. Reconstruction Quality — Baseline at Different Layers")
code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
fig.suptitle('LISTA-CP Baseline: Reconstruction at Each Stage', fontsize=11, fontweight='bold')

for ax, (layer, label) in zip(axes, [
        (1,    f'Layer 1  →  NMSE = {nmse_base[0]:.1f} dB'),
        (T//2, f'Layer {T//2}  →  NMSE = {nmse_base[T//2-1]:.1f} dB'),
        (T,    f'Layer {T} (final)  →  NMSE = {nmse_base[-1]:.1f} dB')]):
    xh_np = xhs_base[layer][0].cpu().numpy()
    ax.plot(x_np,  'C0', lw=0.9, alpha=0.55, label='True x')
    ax.plot(xh_np, 'C2', lw=0.9, alpha=0.90, label='Estimated x̂')
    ax.set_title(label, fontsize=9.5)
    ax.set_xlabel('Index n'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout(); plt.show()
""")

# ─── Cell 8: NMSE comparison ─────────────────────────────────────────────────
md("## 7. NMSE Convergence — All Models")
code("""
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(range(1, T+1), nmse_base, 'C0-o', lw=2.2, ms=6,
        label='LISTA-CP Baseline  (all 16 layers)')

ax.plot(range(1, len(nmse_scratch)+1), nmse_scratch, 'C2--s', lw=2.2, ms=6,
        label=f'LISTA-CP-KAN Scratch  ({KAN_SCRATCH_LAYERS}/16 layers trained so far)')
ax.axvline(KAN_SCRATCH_LAYERS + 0.5, color='C2', lw=1, ls=':', alpha=0.6,
           label=f'← training stopped at layer {KAN_SCRATCH_LAYERS}')

ax.plot(range(1, T+1), nmse_ft, 'C3-^', lw=2.2, ms=6,
        label='LISTA-CP-KAN Fine-tuned  (started from trained baseline)')

ax.axhline(-30.0, color='gray', ls='--', lw=1.2, label='-30 dB target')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('NMSE (dB)  [lower = better]', fontsize=12)
ax.set_title('NMSE per Layer — All Three Models', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.35)
ax.set_xticks(range(1, T+1))
plt.tight_layout(); plt.show()

print(f"\\nBaseline final NMSE     : {nmse_base[-1]:.2f} dB")
print(f"KAN Fine-tuned final    : {nmse_ft[-1]:.2f} dB  "
      f"({'better' if nmse_ft[-1] < nmse_base[-1] else 'worse'} than baseline by "
      f"{abs(nmse_ft[-1]-nmse_base[-1]):.2f} dB)")
print(f"KAN Scratch at layer {KAN_SCRATCH_LAYERS}  : {nmse_scratch[-1]:.2f} dB")
""")

# ─── Cell 9: Adaptive threshold ───────────────────────────────────────────────
md("""## 8. Key Contribution — Adaptive vs Fixed Threshold

In LISTA-CP baseline, each layer uses a **single fixed scalar** θ_k — same for every input sample.

In LISTA-CP-KAN, the threshold is **predicted from the residual** — it changes per sample and per layer.
The plot below shows how the two differ on this one test sample.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# ── Left: threshold comparison ────────────────────────────────────────────────
ax = axes[0]
base_thetas = [baseline.thetas[k].item() for k in range(T)]

# KAN scratch thresholds
thresh_scratch = []
kan_scratch.eval()
with torch.no_grad():
    xh = torch.zeros(1, N, device=device)
    for k in range(KAN_SCRATCH_LAYERS):
        W   = kan_scratch.Ws[k]
        res = y_demo - xh @ kan_scratch.A.T
        th  = kan_scratch.predictors[k](res)
        thresh_scratch.append(th.item())
        v   = xh + res @ W.T
        xh  = shrink_free(v, th)

# KAN fine-tune thresholds
thresh_ft = []
kan_ft.eval()
with torch.no_grad():
    xh = torch.zeros(1, N, device=device)
    for k in range(T):
        W   = kan_ft.Ws[k]
        res = y_demo - xh @ kan_ft.A.T
        th  = kan_ft.predictors[k](res)
        thresh_ft.append(th.item())
        v   = xh + res @ W.T
        xh  = shrink_free(v, th)

ax.plot(range(1, T+1), base_thetas, 'C0-o', lw=2, ms=6,
        label='Baseline: fixed θ_k  (same for all samples)')
ax.plot(range(1, len(thresh_scratch)+1), thresh_scratch, 'C2--s', lw=2, ms=6,
        label='KAN Scratch: adaptive θ_k  (this sample)')
ax.plot(range(1, T+1), thresh_ft, 'C3-^', lw=2, ms=6,
        label='KAN Fine-tune: adaptive θ_k  (this sample)')
ax.set_xlabel('Layer', fontsize=11); ax.set_ylabel('Threshold θ', fontsize=11)
ax.set_title('Threshold per Layer', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.35)

# ── Right: zoom on final reconstruction ───────────────────────────────────────
ax = axes[1]
x_base_final    = xhs_base[-1][0].cpu().numpy()
x_scratch_final = xhs_scratch[-1][0].cpu().numpy()
x_ft_final      = xhs_ft[-1][0].cpu().numpy()

# show only first 100 elements so it's readable
idx = np.arange(100)
ax.plot(idx, x_np[:100], 'k', lw=1.2, alpha=0.6, label='True x')
ax.plot(idx, x_base_final[:100], 'C0', lw=1, alpha=0.8,
        label=f'Baseline  ({nmse_base[-1]:.1f} dB)')
ax.plot(idx, x_ft_final[:100], 'C3--', lw=1, alpha=0.8,
        label=f'KAN Fine-tune  ({nmse_ft[-1]:.1f} dB)')
ax.set_xlabel('Index n (first 100)', fontsize=11)
ax.set_ylabel('Amplitude', fontsize=11)
ax.set_title('Final Reconstruction — First 100 Elements', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Adaptive Thresholding: Key Contribution', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()
""")

# ─── Cell 10: Summary table ───────────────────────────────────────────────────
md("## 9. Summary")
code("""
print("=" * 62)
print(f"  {'Model':<28}  {'Params':>8}  {'NMSE (dB)':>10}")
print("=" * 62)

bp = sum(p.numel() for p in baseline.parameters())
print(f"  {'Baseline LISTA-CP':<28}  {bp:>8,}  {nmse_base[-1]:>10.2f}")

ksp = sum(p.numel() for p in kan_scratch.parameters())
print(f"  {'KAN Scratch ('+str(KAN_SCRATCH_LAYERS)+'/16 layers)':<28}  "
      f"{ksp:>8,}  {nmse_scratch[-1]:>10.2f}  ← layer {KAN_SCRATCH_LAYERS} only")

kfp = sum(p.numel() for p in kan_ft.parameters())
print(f"  {'KAN Fine-tuned (all 16)':<28}  {kfp:>8,}  {nmse_ft[-1]:>10.2f}")
print("=" * 62)

print("\\nNotes:")
print(f"  • KAN Scratch still training — only {KAN_SCRATCH_LAYERS}/16 layers done.")
print(f"    Full 16-layer result expected to improve further.")
print(f"  • KAN Fine-tune started from trained baseline weights.")
print(f"    Fine-tuning a converged model is harder — W is at a local minimum.")
print(f"  • KAN Scratch trains W + KAN together from random init (better approach).")
""")

# ─── Cell 11: Image reconstruction intro ─────────────────────────────────────
md("""---
## 10. Image Reconstruction via Compressed Sensing

**How it works on images:**

```
Image patch (20×25 = 500 pixels)
       ↓  2D-DCT  (frequency transform)
DCT coefficients x  (500-dim, sparse — most high-freq = 0)
       ↓  y = A·x  (compress: 500 → 250 measurements)
Measurements y  (250 numbers stored / transmitted)
       ↓  LISTA-CP / KAN  (reconstruct)
Recovered DCT x̂  (500-dim)
       ↓  2D-IDCT
Reconstructed patch  (20×25 pixels)
```

**Why DCT?** Natural images are sparse in the frequency (DCT) domain —
most energy is in low-frequency coefficients. This matches our model's
assumption of sparse signals with pnz ≈ 10%.

**Available images:**
`barbara`, `boats`, `cameraman`, `fingerprint`, `flinstones`,
`foreman`, `house`, `lena256`, `Monarch`, `Parrots`, `peppers256`
""")

code("""
from PIL import Image
from scipy.fft import dct, idct as scipy_idct

# ── Image selection — change this to try different images ────────────────────
IMAGE_NAME = 'cameraman'     # try: barbara, lena256, boats, house, Monarch ...
IMAGE_PATH = f'LISTA-CPSS/data/test_images/{IMAGE_NAME}.tif'

# ── Patch layout ─────────────────────────────────────────────────────────────
PATCH_H, PATCH_W = 20, 25   # 20×25 = 500 = N  (model input size)
PNZ_IMG = 0.10              # keep top 10% DCT coefficients (matches training)
BATCH   = 128               # patches processed at once (increase if you have GPU)

# ── Load + show available images ─────────────────────────────────────────────
import glob
img_files = sorted(glob.glob('LISTA-CPSS/data/test_images/*.tif'))
print("Available images:")
for f in img_files:
    img_tmp = Image.open(f).convert('L')
    print(f"  {os.path.basename(f):25s}  {img_tmp.size[0]}×{img_tmp.size[1]} px")

print(f"\\n→ Selected: {IMAGE_NAME}")
""")

# ─── Cell 12: Image processing helpers ───────────────────────────────────────
md("### Helper functions (run once)")
code("""
def load_gray(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0

def crop_to_patches(img, ph, pw):
    H, W = img.shape
    H2   = (H // ph) * ph
    W2   = (W // pw) * pw
    return img[:H2, :W2]

def extract_patches(img, ph, pw):
    H, W  = img.shape
    nH, nW = H // ph, W // pw
    # shape: (nH*nW, ph*pw) — each row is one flattened patch
    patches = img.reshape(nH, ph, nW, pw).transpose(0,2,1,3).reshape(-1, ph*pw)
    return patches, nH, nW

def reassemble(patches_flat, nH, nW, ph, pw):
    return (patches_flat.reshape(nH, nW, ph, pw)
                        .transpose(0,2,1,3)
                        .reshape(nH*ph, nW*pw))

def apply_dct2(patches):
    # patches: (n_patches, ph*pw)
    ph, pw = PATCH_H, PATCH_W
    p3d  = patches.reshape(-1, ph, pw)
    out  = dct(dct(p3d, axis=1, norm='ortho'), axis=2, norm='ortho')
    return out.reshape(-1, ph*pw)

def apply_idct2(patches):
    ph, pw = PATCH_H, PATCH_W
    p3d  = patches.reshape(-1, ph, pw)
    out  = scipy_idct(scipy_idct(p3d, axis=1, norm='ortho'), axis=2, norm='ortho')
    return out.reshape(-1, ph*pw)

def sparsify(dct_patches, pnz):
    x = dct_patches.copy()
    k = max(1, int(x.shape[1] * pnz))
    for i in range(len(x)):
        row = x[i]
        thresh_idx = np.argsort(np.abs(row))[:-k]
        row[thresh_idx] = 0
        x[i] = row
    return x

def psnr(orig, recon):
    mse = np.mean((orig.astype(float) - recon.astype(float))**2)
    return float('inf') if mse == 0 else 20*np.log10(1.0/np.sqrt(mse))

def reconstruct_patches(model, dct_patches_sparse, A_t, device, bs=128):
    \"\"\"Run model on batches of DCT patches, return reconstructed DCT patches.\"\"\"
    model.eval()
    all_xhat = []
    n = len(dct_patches_sparse)
    with torch.no_grad():
        for start in range(0, n, bs):
            batch = torch.from_numpy(
                dct_patches_sparse[start:start+bs]).float().to(device)
            y_batch = batch @ A_t.T          # compress: (bs, N) → (bs, M)
            xhs     = model(y_batch)
            all_xhat.append(xhs[-1].cpu().numpy())
    return np.vstack(all_xhat)              # (n_patches, N)

print("Helper functions defined.")
print(f"Patch size: {PATCH_H}×{PATCH_W} = {PATCH_H*PATCH_W} pixels = N")
print(f"Compression: {PATCH_H*PATCH_W} → {M} measurements  "
      f"({M/(PATCH_H*PATCH_W):.0%} of original)")
""")

# ─── Cell 13: Run reconstruction ─────────────────────────────────────────────
md("### Run Image Reconstruction (all 3 models)")
code("""
import time

# ── Load image ────────────────────────────────────────────────────────────────
img_orig = load_gray(IMAGE_PATH)
img_crop = crop_to_patches(img_orig, PATCH_H, PATCH_W)
H, W     = img_crop.shape
print(f"Image: {IMAGE_NAME}  |  Original: {img_orig.shape}  |  Cropped: {H}×{W}")

# ── Extract patches + DCT ─────────────────────────────────────────────────────
patches_px, nH, nW = extract_patches(img_crop, PATCH_H, PATCH_W)
dct_patches         = apply_dct2(patches_px)
dct_sparse          = sparsify(dct_patches, PNZ_IMG)

n_patches = len(patches_px)
nnz_avg   = int(np.mean((dct_sparse != 0).sum(axis=1)))
print(f"Patches: {nH}×{nW} = {n_patches} total  |  avg non-zeros: {nnz_avg}/{N}  ({nnz_avg/N:.1%})")

# ── Reconstruct with each model ───────────────────────────────────────────────
print("\\nReconstructing...")

t0 = time.time()
xhat_base = reconstruct_patches(baseline, dct_sparse, A_t, device, BATCH)
img_base  = np.clip(reassemble(apply_idct2(xhat_base), nH, nW, PATCH_H, PATCH_W), 0, 1)
print(f"  Baseline       : {time.time()-t0:.1f}s  |  PSNR = {psnr(img_crop, img_base):.2f} dB")

t0 = time.time()
xhat_ft  = reconstruct_patches(kan_ft, dct_sparse, A_t, device, BATCH)
img_ft   = np.clip(reassemble(apply_idct2(xhat_ft), nH, nW, PATCH_H, PATCH_W), 0, 1)
print(f"  KAN Fine-tuned : {time.time()-t0:.1f}s  |  PSNR = {psnr(img_crop, img_ft):.2f} dB")

t0 = time.time()
xhat_sc  = reconstruct_patches(kan_scratch, dct_sparse, A_t, device, BATCH)
img_sc   = np.clip(reassemble(apply_idct2(xhat_sc), nH, nW, PATCH_H, PATCH_W), 0, 1)
psnr_sc  = psnr(img_crop, img_sc)
print(f"  KAN Scratch    : {time.time()-t0:.1f}s  |  PSNR = {psnr_sc:.2f} dB  "
      f"(only {KAN_SCRATCH_LAYERS}/16 layers)")

# Reference: oracle sparsified (best possible with this sparsity level)
img_oracle = np.clip(reassemble(apply_idct2(dct_sparse), nH, nW, PATCH_H, PATCH_W), 0, 1)
print(f"  Oracle (top-{nnz_avg} DCT, IDCT): PSNR = {psnr(img_crop, img_oracle):.2f} dB")
""")

# ─── Cell 14: Show image results ─────────────────────────────────────────────
md("### Visual Comparison")
code("""
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle(f'Compressed Sensing Image Reconstruction — {IMAGE_NAME}\\n'
             f'Compression: {N}→{M} per patch  ({M/N:.0%} measurements)',
             fontsize=12, fontweight='bold')

imgs_to_show = [
    (img_crop,   'Original\n(cropped)',                           None),
    (img_oracle, f'Oracle sparsified\n(top {nnz_avg} DCT coefs)\nPSNR={psnr(img_crop,img_oracle):.1f}dB', None),
    (img_base,   f'Baseline LISTA-CP\\nPSNR={psnr(img_crop,img_base):.2f} dB', None),
    (img_ft,     f'KAN Fine-tuned\\nPSNR={psnr(img_crop,img_ft):.2f} dB',    None),
    (img_sc,     f'KAN Scratch ({KAN_SCRATCH_LAYERS}/16 layers)\\nPSNR={psnr_sc:.2f} dB', None),
    (np.abs(img_crop - img_base)*5,
                 'Error map\\n(Baseline, ×5)',                    'hot'),
]

for ax, (im, title, cmap) in zip(axes.flat, imgs_to_show):
    kwargs = dict(cmap=cmap or 'gray', vmin=0, vmax=1 if cmap is None else None)
    ax.imshow(im, **kwargs)
    ax.set_title(title, fontsize=9.5)
    ax.axis('off')

plt.tight_layout(); plt.show()
""")

# ─── Cell 15: Patch zoom ─────────────────────────────────────────────────────
md("### Zoom: Single Patch Pipeline")
code("""
# Pick a visually interesting patch (try different indices)
PATCH_IDX = min(30, n_patches - 1)   # change this to pick a different patch

pi, pj = PATCH_IDX // nW, PATCH_IDX % nW
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
fig.suptitle(f'Single Patch Pipeline  (patch #{PATCH_IDX}, position row={pi} col={pj})',
             fontsize=11, fontweight='bold')

orig_patch  = patches_px[PATCH_IDX].reshape(PATCH_H, PATCH_W)
spar_patch  = apply_idct2(dct_sparse[PATCH_IDX:PATCH_IDX+1]).reshape(PATCH_H, PATCH_W)
base_patch  = apply_idct2(xhat_base[PATCH_IDX:PATCH_IDX+1]).reshape(PATCH_H, PATCH_W)
ft_patch    = apply_idct2(xhat_ft[PATCH_IDX:PATCH_IDX+1]).reshape(PATCH_H, PATCH_W)
dct_vis     = np.log1p(np.abs(dct_patches[PATCH_IDX].reshape(PATCH_H, PATCH_W)))

labels = ['Original patch',
          'DCT (log scale)',
          f'Oracle (IDCT of\\nsparse DCT)',
          f'Baseline recon\\nPSNR={psnr(orig_patch,np.clip(base_patch,0,1)):.1f}dB',
          f'KAN Fine-tune\\nPSNR={psnr(orig_patch,np.clip(ft_patch,0,1)):.1f}dB']
images = [orig_patch, dct_vis, spar_patch, base_patch, ft_patch]
cmaps  = ['gray', 'viridis', 'gray', 'gray', 'gray']

for ax, im, lbl, cm in zip(axes, images, labels, cmaps):
    ax.imshow(np.clip(im, 0, None), cmap=cm, interpolation='nearest',
              aspect='auto')
    ax.set_title(lbl, fontsize=9)
    ax.axis('off')

plt.tight_layout(); plt.show()

print(f"Patch size: {PATCH_H}×{PATCH_W} = {N} pixels")
print(f"Compressed to: {M} measurements  ({M/N:.0%} of original)")
print(f"Non-zero DCT coeffs: {int((dct_sparse[PATCH_IDX]!=0).sum())} / {N}  ({PNZ_IMG:.0%})")
""")

# ─── Cell 16: Try different image ─────────────────────────────────────────────
md("""### Try a Different Image

To try another image, just change `IMAGE_NAME` in the **cell above** (cell 10) and re-run cells 10 → 15.

| Image | Description |
|---|---|
| `cameraman` | Classic 256×256 grayscale |
| `lena256` | Classic portrait, 256×256 |
| `barbara` | Texture-rich, 512×512 |
| `boats` | Fine detail, 512×512 |
| `house` | Smooth regions, 256×256 |
| `Monarch` | Butterfly, high detail |
| `fingerprint` | Fine lines/texture |
""")

# ─── Write notebook ──────────────────────────────────────────────────────────
nb.cells = cells
with open('demo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("demo.ipynb created successfully.")
