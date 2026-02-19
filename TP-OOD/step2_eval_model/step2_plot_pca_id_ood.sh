#!/usr/bin/env bash
set -euo pipefail

# Step 2 (PCA): plot ID/OOD feature distribution on first two PCs.
#
# Usage:
#   bash TP-OOD/step2_eval_model/step2_plot_pca_id_ood.sh \
#     [CKPT_PATH] [GPU_ID] [BATCH_SIZE] [ID_SPLIT] [OOD_SPLIT] [OOD_DATASET] [OUTPUT_DIR] [PLOT_MAX_ID_POINTS]
#
# Example:
#   bash TP-OOD/step2_eval_model/step2_plot_pca_id_ood.sh \
#     results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/model_epoch100.ckpt \
#     0 200 test farood all \
#     results/cifar100_resnet18_32x32_base_e100_lr0.1_default/pca 5000

CKPT_PATH="${1:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/model_epoch100.ckpt}"
GPU_ID="${2:-0}"
BATCH_SIZE="${3:-200}"
ID_SPLIT="${4:-test}"
OOD_SPLIT="${5:-farood}"
OOD_DATASET="${6:-all}"
OUTPUT_DIR="${7:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/pca}"
PLOT_MAX_ID_POINTS="${8:-5000}"
PLOT_MAX_OOD_POINTS="${PLOT_MAX_OOD_POINTS:-500}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python TP-OOD/step2_eval_model/plot_pca_id_ood.py \
  --ckpt-path "${CKPT_PATH}" \
  --id-split "${ID_SPLIT}" \
  --ood-split "${OOD_SPLIT}" \
  --ood-dataset "${OOD_DATASET}" \
  --device cuda \
  --num-workers 8 \
  --batch-size "${BATCH_SIZE}" \
  --plot-max-id-points "${PLOT_MAX_ID_POINTS}" \
  --plot-max-ood-points "${PLOT_MAX_OOD_POINTS}"\
  --output-dir "${OUTPUT_DIR}"

echo "PCA plotting finished."
echo "Outputs: ${PROJECT_ROOT}/${OUTPUT_DIR}"
