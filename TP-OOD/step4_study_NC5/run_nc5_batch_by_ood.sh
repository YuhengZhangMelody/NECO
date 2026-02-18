#!/usr/bin/env bash
set -euo pipefail

# Batch NC5 compute+plot for each OOD dataset.
# Usage:
#   bash TP-OOD/step4_study_NC5/run_nc5_batch_by_ood.sh [SEED_DIRS] [EPOCHS] [DEVICE] [NUM_WORKERS] [BATCH_SIZE]
# Example:
#   bash TP-OOD/step4_study_NC5/run_nc5_batch_by_ood.sh s0,s1,s2 10,20,30,40,50,60,70,80,90,100 cuda 8 128

SEED_DIRS="${1:-}"
EPOCHS="${2:-10,20,30,40,50,60,70,80,90,100}"
DEVICE="${3:-cuda}"
NUM_WORKERS="${4:-8}"
BATCH_SIZE="${5:-128}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CKPT_ROOT="results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
OUT_ROOT="${CKPT_ROOT}/nc5_by_ood"
mkdir -p "${OUT_ROOT}"

# split:dataset
TARGETS=(
  "nearood:cifar10"
  "nearood:tin"
  "farood:mnist"
  "farood:svhn"
  "farood:texture"
  "farood:places365"
)

for item in "${TARGETS[@]}"; do
  split="${item%%:*}"
  ds="${item##*:}"
  echo "==== NC5 ${split}/${ds} ===="

  csv_path="${OUT_ROOT}/nc5_${split}_${ds}_by_seed_epoch.csv"
  plot_dir="${OUT_ROOT}/plots_${split}_${ds}"

  python TP-OOD/step4_study_NC5/compute_nc5.py \
    --ckpt-root "${CKPT_ROOT}" \
    --seed-dirs "${SEED_DIRS}" \
    --epochs "${EPOCHS}" \
    --include-best \
    --ood-split "${split}" \
    --ood-dataset "${ds}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    --output-csv "${csv_path}"

  python TP-OOD/step4_study_NC5/plot_nc5_diagnostic.py \
    --csv-path "${csv_path}" \
    --seed-dirs "${SEED_DIRS}" \
    --epochs "${EPOCHS}" \
    --output-dir "${plot_dir}"
done

echo "Done. Outputs in: ${OUT_ROOT}"
