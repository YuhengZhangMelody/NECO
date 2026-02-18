#!/usr/bin/env bash
set -euo pipefail

# Step 3: compute NC1-NC4 for one or multiple datasets.
#
# Usage:
#   bash TP-OOD/step3_study_NC1-4/step3_compute_nc1-4.sh \
#     [DATASETS] [SEED_DIRS] [EPOCHS] [DEVICE] [NUM_WORKERS] [BATCH_SIZE] [OOD_SPLIT] [OOD_DATASET]
#
# Example:
#   bash TP-OOD/step3_study_NC1-4/step3_compute_nc1-4.sh \
#     cifar100,cifar10 s0,s1,s2 20,40,60,80,100 cuda 8 128 farood all

DATASETS="${1:-cifar100}"
SEED_DIRS="${2:-}"
EPOCHS="${3:-10,20,30,40,50,60,70,80,90,100}"
DEVICE="${4:-cuda}"
NUM_WORKERS="${5:-8}"
BATCH_SIZE="${6:-128}"
OOD_SPLIT="${7:-farood}"
OOD_DATASET="${8:-all}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Use env var to override naming if your experiment folders differ.
CKPT_ROOT_TEMPLATE="${CKPT_ROOT_TEMPLATE:-results/{dataset}_resnet18_32x32_base_e100_lr0.1_default}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a DATASET_ARR <<< "${DATASETS}"

for dataset in "${DATASET_ARR[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [ -z "${dataset}" ] && continue

  ckpt_root="${CKPT_ROOT_TEMPLATE//\{dataset\}/${dataset}}"
  output_base="${ckpt_root}/nc1-4_by_ood"
  output_csv="${output_base}/nc1-4_${OOD_SPLIT}_${OOD_DATASET}_by_seed_epoch.csv"

  echo "[step3] dataset=${dataset} ckpt_root=${ckpt_root}"

  python TP-OOD/step3_study_NC1-4/compute_nc1-4.py \
    --dataset "${dataset}" \
    --ckpt-root "${ckpt_root}" \
    --seed-dirs "${SEED_DIRS}" \
    --epochs "${EPOCHS}" \
    --include-best \
    --ood-split "${OOD_SPLIT}" \
    --ood-dataset "${OOD_DATASET}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    --output-csv "${output_csv}"

  echo "[step3] saved: ${PROJECT_ROOT}/${output_csv}"
done
