"""
generate_report.py
Generates a detailed project report as a Word (.docx) file.
Run: python generate_report.py
Requires: pip install python-docx
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    return p


def body(doc, text, bold=False, italic=False, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    return p


def bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    return p


def code_block(doc, text):
    for line in text.strip().split('\n'):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        run = p.add_run(line if line else ' ')
        run.font.name = 'Courier New'
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x8a)
    return


def divider(doc):
    doc.add_paragraph('─' * 80)


def make_table(doc, headers, rows, header_color='1F4E79'):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(cell, header_color)
    # Data rows
    for ri, row in enumerate(rows):
        tr = table.rows[ri + 1]
        for ci, val in enumerate(row):
            tr.cells[ci].text = str(val)
            if ri % 2 == 1:
                set_cell_bg(tr.cells[ci], 'D9E1F2')
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Document
# ─────────────────────────────────────────────────────────────────────────────

doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# Default font
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

# ── Title ────────────────────────────────────────────────────────────────────
title = doc.add_heading('Adaptive Thresholding in LISTA-CP Using KAN and MLP', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run('CS 754 — Advanced Image Processing  |  Project Technical Report')
run.font.size  = Pt(12)
run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

date_p = doc.add_paragraph()
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_p.add_run(f'Date: {datetime.date.today().strftime("%B %d, %Y")}').font.size = Pt(10)

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Problem Statement
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '1. Problem Statement', level=1)

body(doc,
     'Sparse signal recovery is the problem of reconstructing a sparse signal x ∈ ℝᴺ '
     'from a small number of linear measurements y ∈ ℝᴹ (M << N) given as:')

body(doc, '    y  =  A x  +  noise', bold=True)

body(doc,
     'where A ∈ ℝᴹˣᴺ is a known sensing matrix. Classical algorithms such as ISTA '
     '(Iterative Shrinkage-Thresholding Algorithm) solve this via repeated soft-thresholding '
     'but converge slowly. Deep unrolling replaces the fixed iterations with a learned, '
     'finite-depth neural network.')

body(doc,
     'Our goal is to investigate whether replacing the fixed scalar threshold in the '
     'LISTA-CP unrolled network with a learned adaptive threshold — either via a '
     'Kolmogorov-Arnold Network (KAN) or a Multi-Layer Perceptron (MLP) — can improve '
     'reconstruction quality as measured by NMSE.')

doc.add_paragraph()
heading(doc, 'Problem Setup', level=2)
make_table(doc,
           ['Parameter', 'Value', 'Description'],
           [
               ['M', '250', 'Number of measurements'],
               ['N', '500', 'Signal dimension'],
               ['T', '16', 'Number of unrolled layers'],
               ['pnz', '0.10', 'Sparsity (fraction of non-zero entries)'],
               ['λ (lam)', '0.4', 'Regularisation / threshold scale'],
               ['SNR', '∞ (noiseless)', 'Training / primary evaluation condition'],
               ['Batch size', '1024', 'Training batch size'],
               ['A', 'Random Gaussian, normalised rows', 'Sensing matrix (fixed across all runs)'],
           ])

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Baseline — LISTA-CP
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '2. Baseline — LISTA-CP', level=1)

heading(doc, '2.1  Architecture', level=2)
body(doc,
     'LISTA-CP (Learned ISTA with Coupled Weights) is a deep unrolled network where '
     'each of the T=16 layers performs one step of soft-thresholding:')

code_block(doc, '''Layer k forward pass:
  res_k   = y  −  x_k · Aᵀ          # residual in measurement space  (batch, M)
  v_k     = x_k + res_k · W_kᵀ      # gradient step                  (batch, N)
  x_{k+1} = shrink(v_k, θ_k)        # soft-threshold                 (batch, N)

shrink(v, θ) = sign(v) · max(|v| − θ, 0)   # element-wise soft thresholding

Parameters per layer:
  W_k ∈ ℝᴺˣᴹ    (untied weight matrix, N×M = 500×250 = 125,000 params)
  θ_k ∈ ℝ        (scalar threshold, 1 param)

Total baseline parameters: 16 × (125,000 + 1) = 2,000,016''')

body(doc,
     'W_k is initialised as Aᵀ / (1.001 · ‖A‖₂²) and each θ_k is initialised to '
     'λ / ‖A‖₂² scaled to match the ISTA threshold at that layer.')

heading(doc, '2.2  Training Strategy — Greedy Layer-by-Layer', level=2)
body(doc,
     'Training follows the greedy layer-by-layer strategy from the original LISTA-CP paper. '
     'Each layer t is trained in three sequential sub-stages:')

bullet(doc, 'Sub-stage 1: Train only the new layer variables (W_t, θ_t) at lr = 5×10⁻⁴')
bullet(doc, 'Sub-stage 2: Fine-tune all layers 0..t jointly at lr = 5×10⁻⁴ × 0.2 = 1×10⁻⁴')
bullet(doc, 'Sub-stage 3: Fine-tune all layers 0..t jointly at lr = 5×10⁻⁴ × 0.02 = 1×10⁻⁵')
bullet(doc, 'After completing layer t, a lr_multiplier for previously trained layers decays by ×0.3')

body(doc, 'Each sub-stage uses:')
bullet(doc, 'Adam optimiser')
bullet(doc, 'Early stopping: if validation NMSE does not improve for BETTER_WAIT=2000 steps, stop')
bullet(doc, 'Maximum steps per sub-stage: MAXIT = 25,000')
bullet(doc, 'Validation every VAL_STEP = 100 steps')
bullet(doc, 'Pre-generated batch pool of POOL_SIZE = 500 batches (refreshed when exhausted)')
bullet(doc, 'Gradient clipping at max-norm 1.0')

heading(doc, '2.3  Loss Function', level=2)
body(doc,
     'The baseline uses the L2 reconstruction loss targeting only the current layer output '
     '(not all layers simultaneously):')

code_block(doc, 'L = 0.5 × Σᵢ (x̂_i − xᵢ)²    (sum over elements in the batch)')

body(doc,
     'This is an un-normalised L2 loss (not divided by ‖x‖²). It focuses the sub-stage '
     'on improving the current layer output, which prevents interference from layers that '
     'are not yet trained.')

heading(doc, '2.4  Evaluation Metric', level=2)
body(doc, 'All models are evaluated using Normalised Mean Squared Error (NMSE) in dB:')

code_block(doc, 'NMSE (dB) = 10 × log₁₀ (E[ ‖x̂ − x‖² ] / E[ ‖x‖² ])')

body(doc,
     'Lower (more negative) NMSE is better. A result of -41.87 dB means the average '
     'squared reconstruction error is ~6.5×10⁻⁵ times the signal power — extremely '
     'accurate recovery.')

heading(doc, '2.5  Baseline Results', level=2)
body(doc, 'The baseline was trained on a Kaggle T4 GPU. Training completed in ~149 minutes.')
body(doc, 'Final result:', bold=True)
bullet(doc, 'Layer 16 NMSE: −41.87 dB (noiseless, on test set)')
bullet(doc, 'Checkpoint saved as: lista_cp_layer16.pth (also aliased to lista_cp_baseline.pth)')
bullet(doc, 'Total parameters: 2,000,016  (16 × W_k + 16 × scalar θ_k)')

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Motivation for Adaptive Thresholding
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '3. Motivation for Adaptive Thresholding', level=1)

body(doc,
     'In classical ISTA, the threshold is fixed across all samples and set analytically. '
     'In the baseline LISTA-CP, the threshold θ_k at each layer is a scalar learned during '
     'training — it is the same for every input sample at inference time.')

body(doc,
     'However, the optimal threshold should intuitively depend on the signal properties. '
     'A sample with a large residual (poor reconstruction so far) might need a larger '
     'threshold, while a sample near convergence might need a smaller one. The fixed scalar '
     'is a compromise across all samples.')

body(doc, 'This motivated two questions:')
bullet(doc, 'Can a KAN learn a nonlinear function from residual statistics → adaptive threshold?')
bullet(doc, 'Can an MLP produce a per-element (rather than per-sample scalar) threshold, '
             'allowing different thresholds for each of the N=500 dimensions?')

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Approach 1: LISTA-CP-KAN (Fine-Tuning)
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '4. Approach 1 — LISTA-CP-KAN via Fine-Tuning (FAILED)', level=1)

heading(doc, '4.1  KAN Architecture', level=2)
body(doc,
     'Kolmogorov-Arnold Networks (KANs) replace fixed linear weights with learnable '
     '1D spline functions on each edge of the network. The KANThetaPredictor takes '
     'norm statistics of the residual as input and outputs a per-sample scalar threshold.')

body(doc, 'Two variants were designed:', bold=True)

body(doc, '1norm variant — Single input feature:')
code_block(doc, 'Input feature: log1p( ‖res_k‖₂ / √M )   →  shape (batch, 1)\nKAN width: [1, 2, 1],  grid=5,  k=3 (cubic splines)')

body(doc, '3norm variant — Three input features:')
code_block(doc, '''Input features: [ log1p(‖res_k‖₁ / M),  log1p(‖res_k‖₂ / √M),  log1p(‖res_k‖∞) ]
                  → shape (batch, 3)
KAN width: [3, 6, 1],  grid=5,  k=3 (cubic splines)''')

body(doc, 'Output pipeline for both variants:')
code_block(doc, '''features = compute_norms(res_k)                    # norm statistics
features = clamp(features, min=-2.0, max=4.0)      # stay within spline grid
raw      = KAN(features)                            # spline network
raw      = raw + theta_bias                         # learnable additive bias
raw      = clamp(raw, min=-10.0, max=5.0)           # prevent runaway
theta    = softplus(raw)                            # (batch, 1), guaranteed > 0''')

body(doc,
     'The log1p transform was critical: residual norms shrink by orders of magnitude from '
     'layer 1 to layer 16. Without log-scaling, late-layer inputs fall far outside the '
     'spline grid range [-2, 4], causing extrapolation artifacts and gradient explosion.')

heading(doc, '4.2  Warm-Start Initialisation', level=2)
body(doc,
     'To begin fine-tuning close to the trained baseline (avoiding catastrophic forgetting), '
     'each KAN predictor was warm-started via initialise_to_constant(θ_k):')

bullet(doc, 'All spline coefficients zeroed → spline outputs 0 for any input')
bullet(doc, 'scale_base zeroed → SiLU base path disabled')
bullet(doc, 'A calibration forward pass measures any floating-point residual from the KAN')
bullet(doc, 'theta_bias set to inv_softplus(θ_k) − kan_residual so that forward() returns '
             'exactly the baseline scalar threshold θ_k for any input')
bullet(doc, 'W_k matrices copied directly from the trained baseline')

heading(doc, '4.3  Fine-Tuning Strategy', level=2)
body(doc, 'Two-phase training was designed:')

body(doc, 'Phase A — KAN Warmup (W frozen):')
bullet(doc, '5 epochs at lr_KAN = 1×10⁻⁷')
bullet(doc, 'W_k frozen to prevent disturbing the well-trained baseline')
bullet(doc, 'After warmup, KAN spline grids updated from the actual residual-norm '
             'distribution seen during inference (better resolution where inputs land)')

body(doc, 'Phase B — Joint Fine-Tuning:')
bullet(doc, '100 epochs, Adam optimiser with two parameter groups')
bullet(doc, 'W_k: lr = 1×10⁻⁶  (very conservative — W already well-trained)')
bullet(doc, 'KAN predictors: lr = 1×10⁻⁶, weight_decay = 1×10⁻⁴')
bullet(doc, 'L1 regularisation on spline coefficients: REG_LAMBDA = 1×10⁻³')
bullet(doc, 'AMP (Automatic Mixed Precision) + GradScaler for stability')
bullet(doc, 'Per-batch NaN skip: if loss is non-finite, zero gradients and continue')
bullet(doc, 'Gradient clipping at max-norm 1.0')
bullet(doc, 'Loss function: NMSE loss summed across all T layers, divided by T')

body(doc, 'Loss function in Phase B:')
code_block(doc, '''L = (1/T) × Σ_{k=1}^{T}  E[ ‖x̂_k − x‖² / ‖x‖² ]

nmse_loss(x̂, x) = mean( sum((x̂ − x)², dim=1) / (sum(x², dim=1) + 1e-10) )
total_loss = sum(nmse_loss(x̂_k, x) for each layer k) / T
           + REG_LAMBDA × mean(|spline_coefs|)''')

heading(doc, '4.4  Results and Failure Analysis', level=2)
body(doc, 'NMSE steadily declined from the first epoch of Phase B:', bold=True, color=(180, 0, 0))

make_table(doc,
           ['Epoch', 'Val NMSE', 'Change from warm-start'],
           [
               ['0 (warm-start)', '≈ −41.87 dB', '—'],
               ['1',             '< −41.87 dB',  'Declining'],
               ['10',            'Worsening',     'Consistent decline'],
               ['38',            'NaN',           'Explosion (spline overflow)'],
           ],
           header_color='8B0000')

body(doc, 'Root cause analysis:')
bullet(doc,
       'W is at a sharp local minimum after greedy layer-by-layer training. Any joint '
       'update to both W and KAN disturbs this minimum, and the loss landscape pushes '
       'W away from the optimal point before KAN can compensate.')
bullet(doc,
       'The KAN spline inputs exceeded the grid range despite log1p normalisation — '
       'at layer 38, a runaway spline produced infinite gradients. Fix: clamped features '
       'to [-2, 4] (the spline grid range) in forward().')
bullet(doc,
       'Even after NaN fixes (clamping, L1 reg, lower lr), the NMSE continued to '
       'decline from the warm-start. The fundamental issue was fine-tuning a converged baseline.')

body(doc,
     'Conclusion: Fine-tuning KAN from a converged baseline does not work. '
     'W and the adaptive threshold need to co-evolve from random initialisation.',
     bold=True, italic=True)

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Approach 2: LISTA-CP-MLP (Fine-Tuning)
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '5. Approach 2 — LISTA-CP-MLP via Fine-Tuning (FAILED)', level=1)

heading(doc, '5.1  MLP Architecture', level=2)
body(doc,
     'Unlike the KAN predictor which outputs a per-sample scalar threshold (batch, 1), '
     'the MLP predictor outputs a per-element threshold (batch, N=500) — a different '
     'threshold for each of the N dimensions, for each sample.')

body(doc, 'Input: the pre-threshold signal v_k (not the residual res_k):')
code_block(doc, '''MLPThetaPredictor architecture:
  v_k  →  LayerNorm(N=500)
       →  Linear(500, 128)
       →  ReLU
       →  Linear(128, 500)
       →  + theta_bias          (learnable per-element bias, shape N)
       →  clamp(−10, +5)
       →  softplus
       →  θ_k                   (batch, 500), per-element threshold > 0

Parameters per predictor:
  Linear(500, 128): 500×128 + 128 = 64,128
  Linear(128, 500): 128×500 + 500 = 64,500
  LayerNorm weights: 500 + 500 = 1,000
  theta_bias:        500
  Total:             130,128 per layer
  16 layers total:   2,082,048 MLP params''')

body(doc,
     'Using v_k (pre-threshold signal) rather than res_k as input allows the MLP to '
     'see per-element information and set per-element thresholds, potentially enabling '
     'element-wise adaptive shrinkage that the scalar KAN cannot achieve.')

heading(doc, '5.2  Warm-Start and Training', level=2)
body(doc,
     'Warm-start: all Linear layer weights and biases were zeroed so the MLP output is 0. '
     'Then theta_bias was set to inv_softplus(θ_k) so forward() returns exactly the '
     'baseline scalar θ_k for all elements and any input.')

body(doc, 'Joint fine-tuning:')
bullet(doc, 'W_k: lr = 1×10⁻⁶')
bullet(doc, 'MLP predictors: lr = 1×10⁻⁴  (MLP more stable than KAN splines)')
bullet(doc, '100 epochs × 100 batches per epoch, batch size 1024')
bullet(doc, 'Cosine annealing LR schedule, AMP, gradient clipping at 1.0')
bullet(doc, 'Loss: NMSE summed across all layers / T')

heading(doc, '5.3  Results and Failure', level=2)
body(doc, 'NMSE declined from the first epoch — same root cause as KAN fine-tuning:', bold=True, color=(180, 0, 0))

bullet(doc, 'The W matrices are at a local minimum; joint updates push them away.')
bullet(doc, 'The MLP warm-start means gradient flow initially tries to activate MLP '
             'parameters while simultaneously perturbing W — unstable combination.')
bullet(doc, 'MLP is more stable numerically than KAN (no NaN explosion), but still shows '
             'monotonic decline in NMSE from epoch 1.')

body(doc,
     'Conclusion: The fine-tuning failure is architectural, not a hyperparameter problem. '
     'It confirms that W and adaptive thresholds must be jointly trained from scratch.',
     bold=True, italic=True)

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Approach 3: LISTA-CP-KAN from Scratch (Current)
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '6. Approach 3 — LISTA-CP-KAN Trained from Scratch (In Progress)', level=1)

heading(doc, '6.1  Motivation', level=2)
body(doc,
     'The fine-tuning experiments established a clear principle: W and the adaptive '
     'threshold predictor must co-evolve together. The solution is to apply the same '
     'greedy layer-by-layer training strategy used for the baseline, but replacing '
     'the scalar θ_k with the KAN predictor from layer 1.')

body(doc,
     'This allows W_k and predictor_k to jointly find a good local minimum from random '
     'initialisation, rather than trying to perturb a tightly optimised W.')

heading(doc, '6.2  Architecture', level=2)
body(doc,
     'Same as LISTA-CP-KAN described in Section 4.1, with the 3norm variant ([3,6,1] KAN) '
     'currently training. The W_k matrices are initialised to Aᵀ / (1.001 · ‖A‖₂²) '
     'and KAN predictors are initialised with default pyKAN random initialisation '
     '(no warm-start from a baseline, since there is no baseline to warm-start from).')

heading(doc, '6.3  Training Strategy', level=2)
body(doc, 'Same greedy 3-sub-stage scheme as the baseline, adapted for the KAN model:')

body(doc, 'For each layer t = 0 … 15:')
bullet(doc, 'Sub-stage 1: Train only W_t + predictor_t parameters  (new layer only)  at lr = 5×10⁻⁴')
bullet(doc, 'Sub-stage 2: Train all layers 0..t  at lr = 5×10⁻⁴ × 0.2 = 1×10⁻⁴')
bullet(doc, 'Sub-stage 3: Train all layers 0..t  at lr = 5×10⁻⁴ × 0.02 = 1×10⁻⁵')
bullet(doc, 'After completing layer t: lr_multiplier decays by ×0.3 for all trained vars')

body(doc, 'Each sub-stage:')
bullet(doc, 'Max 25,000 steps with early stopping (BETTER_WAIT = 2,000 steps)')
bullet(doc, 'Best model checkpoint saved per sub-stage')
bullet(doc, 'Gradient clipping max-norm = 1.0')
bullet(doc, 'L1 spline regularisation: REG_LAMBDA = 1×10⁻⁴')

body(doc, 'Loss function — L2 reconstruction loss targeting current layer only:')
code_block(doc, 'L = 0.5 × Σᵢ (x̂_{layer_t, i} − xᵢ)²  +  REG_LAMBDA × mean(|spline_coefs|)')

body(doc, 'Resume logic:')
bullet(doc, 'Checkpoint saved after each complete layer: lista_cp_kan_scratch_layer{k:02d}.pth')
bullet(doc, 'On restart, scans layers 16→1 for the latest completed checkpoint')
bullet(doc, 'Reconstructs lr_multiplier history for all completed layers from DECAY_RATE^(layers_since_k)')

heading(doc, '6.4  Current Status (as of training session)', level=2)
body(doc, 'Variant being trained: 3norm ([3, 6, 1] KAN, 3 input features)')

make_table(doc,
           ['Metric', 'Value'],
           [
               ['Variant', '3norm ([3,6,1] KAN)'],
               ['GPU', 'Kaggle T4 ×2 (single GPU used — KAN is CPU-bound)'],
               ['Session elapsed', '~9 hours'],
               ['Current layer', '5 / 16'],
               ['Val NMSE at step 20000', '−20.69 dB (layer 5 still training)'],
               ['Estimated total time', '~20–25 hours across multiple sessions'],
               ['GPU utilisation', '~36% (bottleneck: pyKAN CPU-bound spline evaluation)'],
               ['CPU utilisation', '~101%'],
           ])

body(doc,
     'Note: pyKAN (v0.2.x) evaluates B-spline basis functions in Python loops, which is '
     'CPU-bound and cannot be accelerated by switching to a different GPU. The P100 GPU '
     'would give ~30% speedup on GPU ops but less than 10% overall improvement.')

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Technical Fixes Applied During Development
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '7. Technical Challenges and Fixes', level=1)

make_table(doc,
           ['Issue', 'Root Cause', 'Fix Applied'],
           [
               ['NaN explosion at epoch 38',
                'KAN spline inputs exceeded grid range [-2,4] → extrapolation → inf gradients',
                'clamp(features, -2.0, 4.0) before KAN forward; log1p normalisation of norms'],
               ['NaN at epoch 1 with [3,6,1] arch',
                'Deeper KAN network more numerically sensitive at very start of training',
                'Per-batch NaN skip: if not isfinite(loss): zero_grad; continue'],
               ['NMSE declining from epoch 1 (fine-tuning)',
                'W at sharp local minimum; any joint update disturbs it',
                'Train from scratch instead of fine-tuning'],
               ['best_state = None at NaN epoch 1',
                'No valid checkpoint was saved before NaN hit, restore was no-op',
                'Save pre-Phase-B state backup; restore if post-B NMSE > pre-B + 1 dB'],
               ['torch.compile key mismatch on checkpoint load',
                'torch.compile wraps model adding _orig_mod. prefix to all keys',
                'get_base_model() unwraps both DataParallel and _orig_mod'],
               ['Kaggle runtime reset losing all progress',
                'Session killed after 12h or page refresh; /kaggle/working is volatile',
                'Per-layer checkpoints with resume scanning highest→lowest layer'],
               ['GPU underutilised (36%)',
                'pyKAN B-spline evaluation is CPU-bound (Python loops)',
                'Accepted as architectural limitation; no change to KAN internals'],
               ['DataParallel overhead slower than single GPU',
                'Small KAN model; cross-GPU synchronisation overhead dominates',
                'Hardcoded n_gpus=1 in training scripts'],
           ])

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '8. Model Architecture Comparison', level=1)

make_table(doc,
           ['Model', 'Threshold Type', 'Input to Predictor', 'Output Shape', 'Extra Params'],
           [
               ['LISTA-CP (Baseline)',  'Scalar θ_k (fixed per layer)', '—', '(1,)', '16 scalars total'],
               ['LISTA-CP-KAN 1norm',   'Per-sample scalar, KAN-predicted',
                'log1p(‖res‖₂/√M), shape (batch,1)', '(batch, 1)', '~4,000 per layer'],
               ['LISTA-CP-KAN 3norm',   'Per-sample scalar, KAN-predicted',
                '[log1p(L1/M), log1p(L2/√M), log1p(L∞)], shape (batch,3)', '(batch, 1)', '~18,000 per layer'],
               ['LISTA-CP-MLP',         'Per-element, MLP-predicted',
                'v_k (pre-threshold signal), shape (batch,500)', '(batch, 500)', '130,128 per layer'],
           ])

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary of Results
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '9. Summary of Results', level=1)

make_table(doc,
           ['Model', 'Training Strategy', 'Layer 16 NMSE (noiseless)', 'Status'],
           [
               ['LISTA-CP Baseline', 'Greedy layer-by-layer from scratch', '−41.87 dB', 'Complete'],
               ['LISTA-CP-KAN 1norm (fine-tune)', 'Phase A+B from baseline',
                'Declining from −41.87 dB', 'Abandoned — degraded'],
               ['LISTA-CP-KAN 3norm (fine-tune)', 'Phase A+B from baseline',
                'NaN at epoch 38 → declining', 'Abandoned — degraded'],
               ['LISTA-CP-MLP (fine-tune)', 'Joint fine-tune from baseline',
                'Declining from −41.87 dB', 'Abandoned — degraded'],
               ['LISTA-CP-KAN 3norm (scratch)', 'Greedy layer-by-layer from scratch',
                'TBD — currently at layer 5', 'In progress (~layer 5/16)'],
               ['LISTA-CP-KAN 1norm (scratch)', 'Greedy layer-by-layer from scratch',
                'TBD — not yet started', 'Pending'],
               ['LISTA-CP-MLP (scratch)', 'Greedy layer-by-layer from scratch',
                'TBD — not yet started', 'Pending'],
           ])

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 10. Key Findings
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '10. Key Findings', level=1)

bullet(doc,
       'Fine-tuning adaptive thresholds on a converged LISTA-CP baseline is fundamentally '
       'unstable — the W matrices are at a sharp local minimum and cannot tolerate joint '
       'parameter updates with a newly introduced threshold predictor.')

bullet(doc,
       'The greedy layer-by-layer training strategy that works for the baseline must also '
       'be applied to the adaptive threshold models. W and the predictor must co-evolve '
       'from random initialisation.')

bullet(doc,
       'KAN spline stability requires: (1) log1p normalisation of input norms to compress '
       'dynamic range across layers; (2) hard clamping of features to the spline grid range; '
       '(3) L1 regularisation on spline coefficients; (4) per-batch NaN skip.')

bullet(doc,
       'pyKAN is CPU-bound — GPU utilisation is only ~36% during KAN training, making '
       'GPU upgrades ineffective for accelerating training. This is a known limitation '
       'of the current pyKAN implementation.')

bullet(doc,
       'A 0.3–0.5 dB improvement over the −41.87 dB baseline would be a meaningful result '
       'in this domain, as the signal is already reconstructed to within ~6.5×10⁻⁵ relative error.')

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 11. Next Steps
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '11. Next Steps', level=1)

bullet(doc, 'Complete KAN 3norm scratch training (layers 6–16) across multiple Kaggle sessions')
bullet(doc, 'Run KAN 1norm scratch training')
bullet(doc, 'Optionally: run MLP scratch training (train_mlp_scratch.py, not yet written)')
bullet(doc, 'Run full evaluation (evaluate.py) comparing all models across 4 SNR conditions')
bullet(doc, 'Analyse KAN adaptivity: std(theta) across test samples per layer')
bullet(doc, 'Visualise learned KAN splines at layer 8')
bullet(doc, 'Produce NMSE-per-layer plots for the report/paper')

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 12. File Structure
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '12. Code and File Structure', level=1)

make_table(doc,
           ['File', 'Purpose'],
           [
               ['problem_setup.py', 'Sensing matrix generation, data sampling, NMSE computation'],
               ['lista_cp_torch.py', 'Baseline LISTA-CP model (LISTACPTorch)'],
               ['train_baseline.py', 'Greedy layer-by-layer training of baseline'],
               ['kan_theta_predictor.py', 'KANThetaPredictor module (1norm and 3norm)'],
               ['lista_cp_kan.py', 'LISTA-CP-KAN model (LISTACPKan)'],
               ['train_kan.py', 'Fine-tuning KAN from baseline (Phase A+B)'],
               ['train_kan_scratch.py', 'Greedy from-scratch training of LISTA-CP-KAN'],
               ['mlp_theta_predictor.py', 'MLPThetaPredictor module (per-element thresholding)'],
               ['lista_cp_mlp.py', 'LISTA-CP-MLP model (LISTACPMlp)'],
               ['train_mlp.py', 'Fine-tuning MLP from baseline'],
               ['evaluate.py', 'Full evaluation: NMSE curves, tables, adaptivity, plots'],
               ['lista_cp_layer16.pth', 'Trained baseline checkpoint (W + A + config)'],
               ['lista_cp_kan_scratch_layer*.pth', 'Per-layer KAN scratch checkpoints for resume'],
           ])

doc.add_paragraph()
body(doc,
     'Note: lista_cp_layer16.pth is the single file needed to share the trained baseline '
     'model. It contains model_state_dict, sensing matrix A, and config (M, N, T, lam, pnz).',
     italic=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
output_path = 'project_report.docx'
doc.save(output_path)
print(f"Report saved to: {output_path}")
