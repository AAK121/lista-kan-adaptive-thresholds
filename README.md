# Adaptive Thresholds for Unrolled Sparse Recovery: LISTA-CP with KAN and MLP

A from-scratch PyTorch reimplementation of **LISTA-CP**, a learned unrolled ISTA network for compressed sensing, that reaches **−42.6 dB NMSE**. On top of it, a study of whether replacing each layer's fixed scalar threshold with an **input-dependent threshold predicted by a Kolmogorov-Arnold Network (KAN) or an MLP** improves recovery.

![PyTorch](https://img.shields.io/badge/PyTorch-algorithm%20unrolling-EE4C2C?logo=pytorch)
![KAN](https://img.shields.io/badge/pykan-Kolmogorov--Arnold%20Networks-6f42c1)

## Overview

**Sparse recovery:** reconstruct a sparse signal x ∈ ℝ⁵⁰⁰ (about 10% non-zeros) from M = 250 linear measurements y = Ax + noise, with a fixed random Gaussian sensing matrix A.

- **ISTA** solves this with repeated gradient steps and soft-thresholding, but needs hundreds of iterations.
- **LISTA-CP** unrolls 16 iterations into a network and learns a weight matrix Wₖ and a threshold θₖ for each layer. It converges orders of magnitude faster.

**The research question.** In LISTA-CP, θₖ is one learned scalar per layer, shared by every input. Classical analysis suggests the ideal threshold depends on the current residual, so a predictor θₖ = g(residual) might adapt to each sample.

What makes it hard:
- The baseline is already very accurate: about 6.5 × 10⁻⁵ relative squared error. A gain of 0.3–0.5 dB would be meaningful.
- Residual norms shrink by orders of magnitude across 16 layers, which pushes KAN spline inputs off their grid.
- Greedy layer-wise training leaves the weights in a sharp minimum that is easy to knock out of.

## Method

```
x̂₀ = 0
for k = 1 … 16:                                    (one unrolled ISTA step per layer)
    r  = y − A x̂ₖ₋₁                                 residual
    v  = x̂ₖ₋₁ + Wₖ r                                learned 500×250 matrix
    θₖ = scalar (baseline)                          ← LISTA-CP
       | KAN([log‖r‖₁, log‖r‖₂, log‖r‖∞])           ← LISTA-CP-KAN: per-sample threshold
       | MLP(LayerNorm(v))                          ← LISTA-CP-MLP: per-coordinate threshold
    x̂ₖ = sign(v) · max(|v| − θₖ, 0)                 soft threshold
```

- **Baseline (`lista_cp_torch.py`, `train_baseline.py`).** A PyTorch reimplementation of Chen et al. (NeurIPS 2018): 16 layers and 2,000,016 parameters. It is trained with the paper's greedy layer-by-layer scheme: for each layer, train the new layer, then fine-tune everything at 0.2× and 0.02× LR, with early stopping. That is 48 stages in total.
- **KAN threshold predictor (`kan_theta_predictor.py`).** pykan networks of width [1,2,1] (one residual norm) or [3,6,1] (L1, L2 and L∞ norms). Inputs are log-normalized and clamped to the spline grid, and a softplus output keeps θ > 0. An `initialise_to_constant(c)` routine calibrates the predictor to reproduce the baseline's θₖ exactly, so a model starts identical to the trained baseline.
- **MLP threshold predictor (`mlp_theta_predictor.py`).** A 2-layer MLP that predicts a separate threshold for each of the 500 coordinates.
- **Training strategies compared:**
  1. Fine-tune KAN on the converged baseline (KAN warm-up, then joint).
  2. Fine-tune MLP on the converged baseline.
  3. Train LISTA-CP-KAN from scratch with the same greedy scheme, so that Wₖ and the predictor co-evolve.
- **Evaluation (`evaluate.py`).**
  - Per-layer NMSE at four noise levels (noiseless, 40, 30 and 20 dB SNR).
  - The number of layers needed to reach −30 dB.
  - An **adaptivity check**: the spread of predicted θ across samples, which catches a predictor that has collapsed to a constant.

## Results

NMSE = 10·log₁₀(‖x̂ − x‖² / ‖x‖²) on the standard 1,000-sample test set from the LISTA-CPSS authors. Lower is better.

| Model | NMSE (layer 16) | Status |
| --- | --- | --- |
| **LISTA-CP baseline (this reimplementation)** | **−42.61 dB** (re-evaluated); −41.87 dB in the project report | trained, ~149 min on a Kaggle T4 |
| LISTA-CP + KAN, fine-tuned from baseline | NMSE got worse from epoch 1 | failed |
| LISTA-CP + MLP, fine-tuned from baseline | NMSE got worse from epoch 1 | failed |
| LISTA-CP-KAN [3,6,1], greedy from scratch | −20.69 dB validation at layer 5 (in training) | partial: checkpoints up to layer 7 of 16 |

Baseline NMSE by depth, re-evaluated with the included `lista_cp_layer16.pth`:

| Layer | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- |
| NMSE (dB) | −12.07 | −26.45 | −36.41 | **−42.61** |

Baseline robustness to measurement noise, from `python evaluate.py` with the included checkpoint:

| Measurement SNR | noiseless | 40 dB | 30 dB | 20 dB |
| --- | --- | --- | --- | --- |
| NMSE (dB) | **−42.61** | −34.70 | −25.30 | −15.94 |

The baseline first reaches −30 dB at layer 10.

The −42.61 dB figure comes from running the included checkpoint through the repository's own `compute_nmse_db_torch` on CPU in float32. The −41.87 dB figure is the one recorded in `docs/project_report.docx`.

Parameter counts:

| Component | Parameters |
| --- | --- |
| Baseline | 2,000,016 (16 × (500×250 + 1)) |
| KAN predictor, 1-norm / 3-norm | ~4k / ~18k per layer |
| MLP predictor | 130,128 per layer |

## Findings

- **Fine-tuning an adaptive threshold onto a converged LISTA-CP fails, and the cause is structural.** Both the KAN and MLP variants started out numerically identical to the baseline, yet NMSE declined from the first epoch. Greedy training leaves W in a sharp minimum, and any joint perturbation pushes it out faster than the new predictor can compensate. Tuning hyperparameters cannot fix that, which motivated co-training from scratch.
- **KAN numerical stability needed several fixes:**
  - NaNs at epoch 38 came from residual norms leaving the spline grid. Log normalization plus clamping fixed it.
  - NaNs at epoch 1 in the deeper [3,6,1] KAN were handled by skipping non-finite batches.
  - The model is restored to its pre-joint-training state if joint training regresses by more than 1 dB.
- **The bottleneck is pykan, not the GPU.** KAN training ran at about 36% GPU utilization because pykan evaluates B-splines in CPU-bound Python loops. DataParallel was measured and disabled: for a 2M-parameter model, gradient synchronization cost more than it saved.
- **Built for ephemeral compute.** Per-layer checkpoints and analytic reconstruction of the learning-rate schedule let the 20-hour-plus from-scratch runs resume across 12-hour Kaggle and Colab sessions.

## Repository structure

```
.
├── problem_setup.py            # sensing matrix, sample generation, test-set loading, NMSE
├── lista_cp_torch.py           # LISTA-CP baseline
├── kan_theta_predictor.py      # KAN threshold predictor (1norm / 3norm), constant initialisation
├── lista_cp_kan.py             # LISTA-CP-KAN
├── mlp_theta_predictor.py      # MLP threshold predictor
├── lista_cp_mlp.py             # LISTA-CP-MLP
├── train_baseline.py           # greedy 3-sub-stage training of the baseline
├── train_kan.py, train_mlp.py  # fine-tuning from the baseline (approaches 1 and 2)
├── train_kan_scratch.py        # greedy from-scratch KAN training (approach 3); *_colab.py variant
├── evaluate.py                 # multi-model, multi-SNR evaluation and adaptivity checks
├── demo.py, demo.ipynb, build_notebook.py   # walkthrough of all models
├── kaggle_eval_cells.py, LISTA_CP_KAN_Kaggle.ipynb, LISTA_CP_KAN_Colab.ipynb
├── generate_report.py          # builds docs/project_report.docx
├── *.pth                       # checkpoints: baseline (16 layers), KAN-scratch (layers 4, 7), KAN fine-tune
├── docs/project_report.docx    # technical report
└── LISTA-CPSS/                 # git submodule: reference implementation + standard test set (VITA-Group)
```

## How to run

```bash
git clone --recursive https://github.com/AAK121/lista-kan-adaptive-thresholds.git
cd lista-kan-adaptive-thresholds
pip install -r requirements.txt          # torch, pykan, numpy, scipy, matplotlib, ...

python train_baseline.py                 # LISTA-CP baseline (resumes from the latest layer checkpoint)
python train_kan.py --variant 3norm      # approach 1: fine-tune KAN thresholds from the baseline
python train_mlp.py                      # approach 2: fine-tune MLP thresholds from the baseline
python train_kan_scratch.py --variant 3norm   # approach 3: greedy from-scratch KAN training

python evaluate.py                       # per-layer NMSE at 4 SNRs, layers-to-−30 dB, adaptivity checks
python demo.py                           # walkthrough plots for the included checkpoints
```

If you cloned without `--recursive`, run `git submodule update --init` to fetch `LISTA-CPSS/`, which holds the test set `data/xtest_n500_p10.npy`.

## Acknowledgements

- **Author:** Alok Kale
- Course project for **CS 754: Advanced Image Processing**, Indian Institute of Technology Bombay.
- The reference TensorFlow implementation and the standard test set come from [VITA-Group/LISTA-CPSS](https://github.com/VITA-Group/LISTA-CPSS). They are included as a submodule and are not modified.
- Kaggle and Google Colab provided GPU time.

## References

1. X. Chen, J. Liu, Z. Wang, W. Yin. *Theoretical Linear Convergence of Unfolded ISTA and its Practical Weights and Thresholds.* NeurIPS 2018.
2. K. Gregor, Y. LeCun. *Learning Fast Approximations of Sparse Coding (LISTA).* ICML 2010.
3. A. Beck, M. Teboulle. *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.* SIAM J. Imaging Sciences 2009.
4. Z. Liu et al. *KAN: Kolmogorov-Arnold Networks.* ICLR 2025.
5. J. Liu, X. Chen, Z. Wang, W. Yin. *ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA.* ICLR 2019.
6. V. Monga, Y. Li, Y. C. Eldar. *Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing.* IEEE Signal Processing Magazine 2021.
