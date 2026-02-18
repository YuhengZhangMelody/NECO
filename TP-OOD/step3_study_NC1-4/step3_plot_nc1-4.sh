#!/usr/bin/env bash
set -euo pipefail

# Step 3: plot NC diagnostics (no slurm).
# Outputs:
# - class_mean_distances_vs_epoch.png
# - within_class_variance_vs_epoch.png
# - cosine_w_vs_means_vs_epoch.png
# - nc1-4_diagnostics_by_seed_epoch.csv
# - nc1-4_diagnostics_aggregate.csv
#
# Usage:
#   bash TP-OOD/step3_study_NC1-4/step3_plot_nc1-4.sh [SEED_DIRS] [EPOCHS] [DEVICE] [NUM_WORKERS] [BATCH_SIZE]
# Example:
#   bash TP-OOD/step3_study_NC1-4/step3_plot_nc1-4.sh s0,s1,s2 10,20,30,40,50,60,70,80,90,100 cuda 8 128

SEED_DIRS="${1:-}"
EPOCHS="${2:-10,20,30,40,50,60,70,80,90,100}"
DEVICE="${3:-cuda}"
NUM_WORKERS="${4:-8}"
BATCH_SIZE="${5:-128}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CHECKPOINTS_ROOT="results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
OUTPUT_DIR="${CHECKPOINTS_ROOT}/plots"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python TP-OOD/step3_study_NC1-4/plot_nc1-4_diagnostics.py \
  --ckpt-root "${CHECKPOINTS_ROOT}" \
  --seed-dirs "${SEED_DIRS}" \
  --epochs "${EPOCHS}" \
  --include-best \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir "${OUTPUT_DIR}"

echo "Plots and tables saved under: ${PROJECT_ROOT}/${OUTPUT_DIR}"
