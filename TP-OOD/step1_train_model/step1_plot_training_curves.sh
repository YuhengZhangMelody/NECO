#!/usr/bin/env bash
set -euo pipefail

# Plot training curves from OpenOOD training log.
# Usage:
#   bash TP-OOD/step2_eval_model/step2_plot_training_curves.sh [LOG_PATH] [OUTPUT_DIR]
# Example:
#   bash TP-OOD/step2_eval_model/step2_plot_training_curves.sh \
#     results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/log.txt \
#     results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/plots

LOG_PATH="${1:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/log.txt}"
OUTPUT_DIR="${2:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/plots}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python TP-OOD/step1_train_model/plot_training_curves.py \
  --log-path "${LOG_PATH}" \
  --output-dir "${OUTPUT_DIR}"

echo "Training curves saved to: ${PROJECT_ROOT}/${OUTPUT_DIR}"
