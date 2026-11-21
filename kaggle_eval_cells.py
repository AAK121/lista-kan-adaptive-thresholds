"""
kaggle_eval_cells.py
Paste each cell block into a Kaggle notebook cell.
Handles pykan version mismatches by detecting KAN width from checkpoint.
"""

# ============================================================
# CELL 1 – install correct pykan version + unzip project
# ============================================================
CELL_1 = r"""
import subprocess, sys, os

# Unzip project into /kaggle/working
ZIP_SRC = '/kaggle/input/cs754project/cs754_research_project.zip'
if os.path.exists(ZIP_SRC):
    subprocess.run(['unzip', '-o', '-q', ZIP_SRC, '-d', '/kaggle/working'], check=True)
    print("Unzipped project.")
else:
    # Try flat zip
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f.endswith('.zip'):
                print(f"Found zip: {os.path.join(root, f)}")

# Pin pykan to the version used during training (0.1.x uses act_fun / coef API)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'pykan==0.1.3', '-q'], check=True)
print("pykan 0.1.3 installed.")
"""

# ============================================================
# CELL 2 – imports and path constants
# ============================================================
CELL_2 = r"""
import os, sys, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dct, idct

# ---- Paths -------------------------------------------------------
# All model files and images are in /kaggle/working/1/
MODEL_DIR = '/kaggle/working/1'

# Add project source to path
for candidate in ['/kaggle/working', '/kaggle/working/1']:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

# ---- Project imports ---------------------------------------------
from problem_setup import random_A, gen_samples_torch, load_test_data, compute_nmse_db_torch
from lista_cp_torch import LISTACPTorch
from lista_cp_kan   import LISTACPKan

print("Imports OK")
print("Files in MODEL_DIR:", [f for f in os.listdir(MODEL_DIR)][:20])
"""

# ============================================================
# CELL 3 – helper: detect KAN width from checkpoint state dict
# ============================================================
CELL_3 = r"""
def detect_kan_width(sd, variant):
    """Read KAN architecture from checkpoint state dict.

    In pykan 0.1.x, act_fun.{i}.coef has shape [in_dim, out_dim, n_coef].
    We reconstruct the full width list by reading each layer's coef shape.
    Returns None if the keys are not present (incompatible checkpoint).
    """
    in_dim = 1 if variant == '1norm' else 3
    prefix = 'predictors.0.kan.act_fun.'
    coef_keys = sorted(
        [k for k in sd if k.startswith(prefix) and k.endswith('.coef')],
        key=lambda k: int(k.replace(prefix, '').split('.')[0])
    )
    if not coef_keys:
        print(f"  [detect_kan_width] No act_fun keys found for variant={variant}. "
              "Keys sample:", list(sd.keys())[:6])
        return None
    width = [in_dim]
    for key in coef_keys:
        shape = sd[key].shape   # [prev_out, this_out, n_coef]
        width.append(int(shape[1]))
    print(f"  Detected KAN width: {width}  (from {len(coef_keys)} act_fun layer(s))")
    return width


def clean_state(sd):
    """Strip DataParallel / torch.compile prefixes."""
    out = {}
    for k, v in sd.items():
        k2 = k
        for pfx in ('module.', '_orig_mod.', 'module._orig_mod.'):
            if k2.startswith(pfx):
                k2 = k2[len(pfx):]
        out[k2] = v
    return out


def p(filename):
    """Resolve a filename to its full path in MODEL_DIR."""
    full = os.path.join(MODEL_DIR, filename)
    if os.path.exists(full):
        return full
    # Try /kaggle/working directly
    alt = os.path.join('/kaggle/working', filename)
    if os.path.exists(alt):
        return alt
    return full  # return even if missing so callers can report the error


print("Helper functions defined.")
"""

# ============================================================
# CELL 4 – load sensing matrix A and test data
# ============================================================
CELL_4 = r"""
M, N, T_LAYERS = 250, 500, 16
LAM  = 0.4
PNZ  = 0.1
SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

np.random.seed(SEED)
A = random_A(M, N, seed=SEED)
A_t = torch.from_numpy(A.astype(np.float32)).to(device)

NPY_PATH = p('xtest_n500_p10.npy')
if os.path.exists(NPY_PATH):
    y_test, x_test = load_test_data(NPY_PATH, A, SNR=np.inf)
    y_t = torch.from_numpy(y_test.astype(np.float32)).to(device)
    x_t = torch.from_numpy(x_test.astype(np.float32)).to(device)
    print(f"Test data loaded: y={y_t.shape}  x={x_t.shape}")
else:
    print(f"WARNING: {NPY_PATH} not found. Generating synthetic test set.")
    from problem_setup import gen_samples_torch
    np.random.seed(SEED + 99)
    y_t, x_t = gen_samples_torch(A_t, 1000, pnz=PNZ, SNR=np.inf, device=device)
    y_t, x_t = y_t.detach(), x_t.detach()
    print(f"Synthetic test: y={y_t.shape}  x={x_t.shape}")

def nmse_per_layer(model, y, x):
    model.eval()
    with torch.no_grad():
        xhs = model(y)
    return [compute_nmse_db_torch(xh, x).item() for xh in xhs[1:]]
"""

# ============================================================
# CELL 5 – load LISTA-CP baseline
# ============================================================
CELL_5 = r"""
# ---------- LISTA-CP baseline ----------
cfg_b = dict(M=M, N=N, T=T_LAYERS, lam=LAM)

def load_baseline(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LISTACPTorch(A, T=cfg_b['T'], lam=cfg_b.get('lam', 0.4)).to(device)
    sd    = clean_state(ckpt['model_state_dict'])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
    model.eval()
    return model

BASELINE_PATH = p('lista_cp_best.pth')
if os.path.exists(BASELINE_PATH):
    baseline = load_baseline(BASELINE_PATH)
    nmse_b   = nmse_per_layer(baseline, y_t, x_t)
    print(f"Baseline LISTA-CP  layer-{T_LAYERS} NMSE: {nmse_b[-1]:.2f} dB")
    print(f"All layers: {[round(v,1) for v in nmse_b]}")
else:
    print(f"Baseline checkpoint not found: {BASELINE_PATH}")
    baseline = None
    nmse_b   = None
"""

# ============================================================
# CELL 6 – load KAN models with auto-detected architecture
# ============================================================
CELL_6 = r"""
cfg_k = dict(M=M, N=N, T=T_LAYERS, lam=LAM)

def load_kan(ckpt_path, label='KAN'):
    if not os.path.exists(ckpt_path):
        print(f"  SKIP {label}: file not found at {ckpt_path}")
        return None, None
    try:
        ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
        variant = ckpt.get('variant', '3norm')
        sd      = clean_state(ckpt['model_state_dict'])

        # Auto-detect KAN width from checkpoint to handle pykan version mismatches
        kan_width = detect_kan_width(sd, variant)
        print(f"  {label}: variant={variant}, kan_width={kan_width}")

        model = LISTACPKan(
            A, T=cfg_k['T'], lam=cfg_k.get('lam', 0.4),
            variant=variant, kan_width=kan_width
        ).to(device)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        size_errs = [k for k in missing + unexpected
                     if 'size mismatch' in str(k).lower()]
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

        model.eval()
        return model, variant
    except Exception as e:
        print(f"  ERROR loading {label}: {e}")
        return None, None

# ---------- KAN fine-tuned (phase A) ----------
KAN_FT_PATH = p('lista_cp_kan3_best.pth')
kan_ft, variant_ft = load_kan(KAN_FT_PATH, 'KAN-FT (3norm)')

# ---------- KAN from scratch ----------
# Use the latest completed layer checkpoint
scratch_ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, 'lista_cp_kan_scratch_layer*.pth')))
if scratch_ckpts:
    latest = scratch_ckpts[-1]
    print(f"Latest scratch checkpoint: {os.path.basename(latest)}")
    kan_sc, variant_sc = load_kan(latest, f'KAN-Scratch ({os.path.basename(latest)})')
else:
    print("No scratch checkpoints found in MODEL_DIR")
    kan_sc, variant_sc = None, None

# Also try the final scratch checkpoint name
SCRATCH_FINAL = p('lista_cp_kan3_scratch.pth')
if os.path.exists(SCRATCH_FINAL) and kan_sc is None:
    kan_sc, variant_sc = load_kan(SCRATCH_FINAL, 'KAN-Scratch final')
"""

# ============================================================
# CELL 7 – Part A: NMSE evaluation table
# ============================================================
CELL_7 = r"""
print("=" * 60)
print("PART A: NMSE Evaluation on Test Set")
print("=" * 60)

results = {}
if baseline is not None:
    results['LISTA-CP'] = nmse_b
if kan_ft is not None:
    results['KAN-FT'] = nmse_per_layer(kan_ft, y_t, x_t)
if kan_sc is not None:
    results['KAN-Scratch'] = nmse_per_layer(kan_sc, y_t, x_t)

print(f"\n{'Layer':>6}", end='')
for name in results:
    print(f"  {name:>12}", end='')
print()
print("-" * (8 + 14 * len(results)))
for i in range(T_LAYERS):
    print(f"{i+1:>6}", end='')
    for name in results:
        v = results[name][i] if i < len(results[name]) else float('nan')
        print(f"  {v:>12.2f}", end='')
    print()

print("\nFinal layer summary:")
for name, vals in results.items():
    print(f"  {name}: {vals[-1]:.2f} dB")
"""

# ============================================================
# CELL 8 – Part A: NMSE curve plot
# ============================================================
CELL_8 = r"""
fig, ax = plt.subplots(figsize=(8, 5))
markers = {'LISTA-CP': 'o-', 'KAN-FT': 's--', 'KAN-Scratch': '^:'}
for name, vals in results.items():
    fmt = markers.get(name, 'x-')
    ax.plot(range(1, len(vals)+1), vals, fmt, label=name, markersize=5)

ax.set_xlabel('Layer')
ax.set_ylabel('NMSE (dB)')
ax.set_title('Per-layer NMSE on Test Set')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/nmse_curves.png', dpi=150)
plt.show()
print("Saved nmse_curves.png")
"""

# ============================================================
# CELL 9 – Part B: Image reconstruction
# ============================================================
CELL_9 = r"""
print("=" * 60)
print("PART B: Image Reconstruction")
print("=" * 60)

PH, PW = 20, 25   # patch size: 20x25 = 500 = N
SPARSITY_K = 0.10  # keep top 10% DCT coefficients

# Find images – check working dir and MODEL_DIR
img_paths = (
    glob.glob('/kaggle/working/*.tif') +
    glob.glob('/kaggle/working/*.png') +
    glob.glob('/kaggle/working/*.jpg') +
    glob.glob(os.path.join(MODEL_DIR, '*.tif')) +
    glob.glob(os.path.join(MODEL_DIR, '*.png')) +
    glob.glob(os.path.join(MODEL_DIR, '*.jpg'))
)
img_paths = sorted(set(img_paths))
print(f"Found {len(img_paths)} image(s): {[os.path.basename(p) for p in img_paths]}")

def reconstruct_image(img_path, model, batch_size=256):
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
    H, W = img.shape

    # Crop to multiples of patch size
    H2 = (H // PH) * PH
    W2 = (W // PW) * PW
    img = img[:H2, :W2]

    nH, nW = H2 // PH, W2 // PW
    patches = img.reshape(nH, PH, nW, PW).transpose(0, 2, 1, 3).reshape(-1, PH * PW)

    # 2-D DCT of each patch → sparse representation
    dct2d = dct(dct(patches.reshape(-1, PH, PW), axis=1, norm='ortho'),
                axis=2, norm='ortho').reshape(-1, N)
    sparse = dct2d.copy()
    k_keep = max(1, int(N * SPARSITY_K))
    for i in range(len(sparse)):
        thresh_idx = np.argsort(np.abs(sparse[i]))[:-k_keep]
        sparse[i, thresh_idx] = 0.0

    # Compress: y = Ax  (A is M×N, x is column vector)
    x_gt = sparse.astype(np.float32)  # (n_patches, N)
    y_meas = (x_gt @ A.T)             # (n_patches, M)  -- A is (M,N), x row vecs

    # Reconstruct in batches
    n_patches = x_gt.shape[0]
    x_rec = np.zeros_like(x_gt)
    model.eval()
    with torch.no_grad():
        for b in range(0, n_patches, batch_size):
            yb = torch.from_numpy(y_meas[b:b+batch_size]).to(device)
            xhs = model(yb)
            x_rec[b:b+batch_size] = xhs[-1].cpu().numpy()

    # IDCT: reconstruct pixel patches
    rec_patches = idct(idct(x_rec.reshape(-1, PH, PW), axis=2, norm='ortho'),
                       axis=1, norm='ortho').reshape(-1, PH * PW)
    gt_patches  = idct(idct(x_gt.reshape(-1, PH, PW), axis=2, norm='ortho'),
                       axis=1, norm='ortho').reshape(-1, PH * PW)

    rec_img = rec_patches.reshape(nH, nW, PH, PW).transpose(0, 2, 1, 3).reshape(H2, W2)
    gt_img  = gt_patches.reshape(nH, nW, PH, PW).transpose(0, 2, 1, 3).reshape(H2, W2)
    rec_img = np.clip(rec_img, 0, 1)

    # PSNR
    mse = np.mean((rec_img - gt_img) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-12 else float('inf')
    return rec_img, gt_img, psnr


def show_reconstruction(img_path, models_dict, save_prefix='recon'):
    name = os.path.splitext(os.path.basename(img_path))[0]
    fig, axes = plt.subplots(1, len(models_dict) + 1,
                             figsize=(4 * (len(models_dict) + 1), 4))
    axes = np.array(axes).flatten()

    # Reference (sparse DCT then IDCT, no reconstruction error)
    img_orig = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
    H2 = (img_orig.shape[0] // PH) * PH
    W2 = (img_orig.shape[1] // PW) * PW
    axes[0].imshow(img_orig[:H2, :W2], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Sparse DCT input')
    axes[0].axis('off')

    psnrs = {}
    for idx, (label, model) in enumerate(models_dict.items(), start=1):
        rec, gt, psnr = reconstruct_image(img_path, model)
        axes[idx].imshow(rec, cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'{label}\nPSNR={psnr:.1f} dB')
        axes[idx].axis('off')
        psnrs[label] = psnr

    plt.suptitle(name, fontsize=12)
    plt.tight_layout()
    out_path = f'/kaggle/working/{save_prefix}_{name}.png'
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  {name}: {psnrs}")
    return psnrs


# Build dict of available models
models_dict = {}
if baseline is not None:
    models_dict['LISTA-CP'] = baseline
if kan_ft is not None:
    models_dict['KAN-FT'] = kan_ft
if kan_sc is not None:
    models_dict['KAN-Scratch'] = kan_sc

if not models_dict:
    print("No models loaded – cannot run reconstruction.")
elif not img_paths:
    print("No images found. Place .tif/.png/.jpg files in /kaggle/working/")
else:
    all_psnrs = {}
    for img_path in img_paths[:5]:   # limit to 5 images
        psnrs = show_reconstruction(img_path, models_dict)
        all_psnrs[os.path.basename(img_path)] = psnrs

    print("\nPSNR Summary:")
    header = f"{'Image':>20}" + "".join(f"  {n:>12}" for n in models_dict)
    print(header)
    for img_name, psnrs in all_psnrs.items():
        row = f"{img_name[:20]:>20}"
        for n in models_dict:
            row += f"  {psnrs.get(n, float('nan')):>12.1f}"
        print(row)
"""

if __name__ == '__main__':
    print("This file contains Kaggle cell code as Python strings.")
    print("Copy each CELL_N string into a separate Kaggle notebook cell.")
    for i in range(1, 10):
        cell = globals().get(f'CELL_{i}', '')
        print(f"\n{'='*60}")
        print(f"CELL {i}")
        print('='*60)
        print(cell.strip())
