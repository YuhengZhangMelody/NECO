#!/usr/bin/env bash
set -euo pipefail

# Step 5: Evaluate NECO (Neural Collapse Inspired OOD Detection) via OpenOOD API.
#
# Usage:
#   bash TP-OOD/step5_neco/step5_eval_ood_neco.sh [ROOT] [GPU_ID] [BATCH_SIZE]
# Example:
#   bash TP-OOD/step5_neco/step5_eval_ood_neco.sh \
#     results/cifar100_resnet18_32x32_base_e100_lr0.1_default 0 200

ROOT="${1:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default}"
GPU_ID="${2:-0}"
BATCH_SIZE="${3:-200}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PP="neco"

echo "==== Evaluating postprocessor: ${PP} ===="
python scripts/eval_ood.py \
  --id-data cifar100 \
  --root "${ROOT}" \
  --postprocessor "${PP}" \
  --batch-size "${BATCH_SIZE}" \
  --save-score \
  --save-csv

echo
echo "Saved: ${ROOT}/s*/scores/${PP}.pkl and ${ROOT}/ood/${PP}.csv"
echo "Step 5 finished."
