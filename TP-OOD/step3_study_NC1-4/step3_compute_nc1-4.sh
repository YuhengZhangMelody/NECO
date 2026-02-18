#!/usr/bin/env bash
set -euo pipefail

# Step 3: compute NC1-NC4 for selected checkpoints on CIFAR-100 ID-train.
# Metric conventions (aligned with neco-mastery/NC_metrics.py):
# - NC1: (1/C) Tr(Sigma_W Sigma_B^dagger)
# - NC2: coherence(M_/||M_||)  [main]
# - NC3: ||W^T/||W^T||_F - M_/||M_||_F||_F^2
# - NC4: NCC mismatch rate
# Extra diagnostic column:
# - nc2_etf_fro: ETF-form NC2 (kept only for reference)
# Usage:
#   bash TP-OOD/step3_study_NC1-4/step3_compute_nc1-4.sh [SEED_DIRS] [EPOCHS] [DEVICE] [NUM_WORKERS] [BATCH_SIZE]
# Example:
#   bash TP-OOD/step3_study_NC1-4/step3_compute_nc1-4.sh s0,s1,s2 20,40,60,80,100 cuda 8 128

SEED_DIRS="${1:-}"
EPOCHS="${2:-20,40,60,80,100}"
DEVICE="${3:-cuda}"
NUM_WORKERS="${4:-8}"
BATCH_SIZE="${5:-128}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Keep checkpoints path configured here.
CHECKPOINTS_ROOT="results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
OUTPUT_CSV="${CHECKPOINTS_ROOT}/nc1-4_by_seed_epoch.csv"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python TP-OOD/step3_study_NC1-4/compute_nc1-4.py \
  --ckpt-root "${CHECKPOINTS_ROOT}" \
  --seed-dirs "${SEED_DIRS}" \
  --epochs "${EPOCHS}" \
  --include-best \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --output-csv "${OUTPUT_CSV}"

echo "NC metrics saved to: ${PROJECT_ROOT}/${OUTPUT_CSV}"
