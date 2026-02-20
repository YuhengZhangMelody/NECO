# TP-OOD (OpenOOD) — Our Implementation README

Author: Yuheng ZHANG, Qizheng WANG
This repository is based on **OpenOOD**. 

You can find our report **"Report-Yuheng_ZHANG_and_Qizheng_WANG.pdf"** directly. Our team code is mainly located under:

- `TP-OOD/` (scripts + analysis code calling OpenOOD)
- `openood/postprocessors/neco_postprocessor.py` (our NECO method integration)
- `configs/postprocessors/neco.yml` (NECO configuration)

The goal of this project is to:
- train an ID classifier (ResNet-18 on CIFAR-100),
- evaluate multiple OOD scoring methods using OpenOOD,
- study Neural Collapse (NC1–NC5),
- implement **NECO** (Neural Collapse Inspired OOD Detection) and evaluate it.

---

## 1) Environment & Setup

We recommend using a conda environment with CUDA support:

    conda create -n openood-cu12 python=3.10 -y
    conda activate openood-cu12

    # install dependencies (example)
    pip install -r requirements.txt
    pip install scikit-learn omegaconf tqdm

### Notes
- Some OpenOOD postprocessors depend on optional libraries (e.g. `libmr`, `statsmodels`).
  To avoid import-time crashes, we modified lazy-loading so evaluation can run without installing those optional dependencies.

---

## 2) Folder Structure (What We Wrote vs OpenOOD)

### OpenOOD code (framework)
Most folders are from OpenOOD, such as:
- `openood/` (models, datasets, evaluators, postprocessors, etc.)
- `configs/` (dataset/network/pipeline/postprocessor configs)
- `scripts/eval_ood.py` (official evaluation entry)

### Our team code (main part)
All project steps are organized under:
- `TP-OOD/step1_train_model/`
- `TP-OOD/step2_eval_model/`
- `TP-OOD/step3_study_NC1-4/`
- `TP-OOD/step4_study_NC5/`
- `TP-OOD/step5_neco/`

This is where we wrote wrappers, analysis scripts, plotting code, and batch runners.

### Our team additions inside OpenOOD
We implemented NECO as a new OpenOOD postprocessor:
- `openood/postprocessors/neco_postprocessor.py` ✅ (written by us)
- `configs/postprocessors/neco.yml` ✅ (written by us)

We also adjusted OpenOOD loading behavior to avoid optional dependency crashes:
- `openood/evaluation_api/postprocessor.py` ✅ (edited by us for lazy import + config compatibility)
- `openood/postprocessors/__init__.py` ✅ (edited by us to avoid importing all postprocessors at import-time)
- (optional) `openood/postprocessors/utils.py` ✅ (edited by us if used elsewhere in the codebase)

---

## 3) Step-by-Step Description

### Step 1 — Train ResNet-18 on CIFAR-100
**Goal:** train the baseline ID model to be used for all OOD experiments.

**Our scripts (written by us):**
- `TP-OOD/step1_train_model/step1_train_cifar100_resnet18.sh`
- `TP-OOD/step1_train_model/plot_training_curves.py` (extract curves from logs and plot)

**How to run:**

    bash TP-OOD/step1_train_model/step1_train_cifar100_resnet18.sh

**Outputs:**
- checkpoints in `results/.../s0/` such as `best.ckpt`
- training logs in `results/.../s0/log.txt`
- plots in `results/.../s0/plots/` (e.g. `training_curves_combined.png`)

---

### Step 2 — Evaluate Standard OOD Scores (OpenOOD)
**Goal:** compare common OOD scoring methods on near-OOD and far-OOD datasets.

**Our scripts (written by us):**
- `TP-OOD/step2_eval_model/step2_eval_ood_5scores.sh`

This script calls OpenOOD’s evaluator multiple times with different postprocessors:
- MSP
- MaxLogit (MLS)
- Mahalanobis (MDS)
- Energy (EBO)
- ViM

**How to run:**

    bash TP-OOD/step2_eval_model/step2_eval_ood_5scores.sh \
      results/cifar100_resnet18_32x32_base_e100_lr0.1_default 0 200

**Outputs:**
- per-sample scores: `results/.../s*/scores/<method>.pkl`
- summary metrics: `results/.../ood/<method>.csv`

---

### Step 3 — Neural Collapse (NC1–NC4)
**Goal:** measure NC1–NC4 along training checkpoints (e.g. every 10 epochs).

**Our scripts (written by us):**
- `TP-OOD/step3_study_NC1-4/compute_nc1-4.py`
- `TP-OOD/step3_study_NC1-4/step3_compute_nc1-4.sh`
- `TP-OOD/step3_study_NC1-4/plot_nc1-4_diagnostics.py`

**What it does:**
- loads intermediate checkpoints
- extracts penultimate-layer features
- computes NC1–NC4 metrics
- saves CSV results and plots

**How to run:**

    bash TP-OOD/step3_study_NC1-4/step3_compute_nc1-4.sh \
      results/cifar100_resnet18_32x32_base_e100_lr0.1_default 0 200

**Outputs:**
- metric CSV files (e.g. `nc1-4_*.csv`)
- plots under `results/.../nc/` (e.g. `nc1-4_vs_epoch_combined.png`)

---

### Step 4 — NC5 (ID/OOD Geometry)
**Goal:** compute NC5, measuring alignment between OOD mean feature and ID class means.

**Our scripts (written by us):**
- `TP-OOD/step4_study_NC5/compute_nc5.py`
- `TP-OOD/step4_study_NC5/run_nc5_batch_by_ood.sh`

**What it does:**
- for each chosen OOD dataset, extract features
- compute NC5 over epochs/checkpoints
- save plots for near/far datasets

**How to run:**

    bash TP-OOD/step4_study_NC5/run_nc5_batch_by_ood.sh \
      results/cifar100_resnet18_32x32_base_e100_lr0.1_default 0 200

**Outputs:**
- NC5 plots under `results/.../nc5/` (e.g. `nc5_vs_epoch.png`)

---

### Step 5 — Implement NECO (Neural Collapse Inspired OOD Detection)
**Goal:** implement NECO and evaluate it like other OpenOOD postprocessors.

#### Step 5.1 — NECO postprocessor (core implementation)
**File written by us:**
- `openood/postprocessors/neco_postprocessor.py`

**Method summary:**
- Extract ID train features `z`.
- Optionally standardize them (StandardScaler).
- Fit PCA on ID train features.
- For each test sample compute:
  `score(x) = ||P_k(z)|| / (||z|| + eps)`
  where `P_k` is projection onto top-k PCA components (`neco_dim`).

#### Step 5.2 — NECO config
**File written by us:**
- `configs/postprocessors/neco.yml`

Typical setting:
- `neco_dim: 100`
- `APS_mode: False` (disabled to avoid automatic search interface differences)

#### Step 5.3 — Step 5 run script
**File written by us:**
- `TP-OOD/step5_neco/step5_eval_ood_neco.sh`

**How to run:**

    bash TP-OOD/step5_neco/step5_eval_ood_neco.sh \
      results/cifar100_resnet18_32x32_base_e100_lr0.1_default 0 200

**Outputs:**
- `results/.../s*/scores/neco.pkl`
- `results/.../ood/neco.csv`

---

## 4) Figures Used in the Report (Overleaf)

We copied selected figures into an Overleaf-friendly layout:

    figures/
      training/training_curves_combined.png
      nc/nc1-4_vs_epoch_combined.png
      nc/w_mu_cos_mean_vs_epoch.png
      nc5/nc5_vs_epoch.png
      pca/pca_id-test_ood-farood-all.png

These are referenced in `main.tex`.

---

## 5) Notes / Troubleshooting

### Optional dependencies issues
Some postprocessors (e.g. OpenMax / AdaScale) require extra packages such as:
- `libmr`
- `statsmodels`

To avoid breaking evaluation when those are not installed, we modified import behavior
so that only the chosen postprocessor is loaded.

### Data download
Some OOD datasets are downloaded automatically the first time you run evaluation
(e.g. `tin`, `places365`, etc.). They are stored under `data/images_classic/`.

---

## 6) Reproducibility Checklist

- [ ] Baseline trained and `best.ckpt` exists
- [ ] Step 2 generates `ood/*.csv` for standard scores
- [ ] Step 3/4 generate NC plots
- [ ] Step 5 generates `ood/neco.csv`
- [ ] Overleaf figures folder contains selected plots
