#!/usr/bin/env bash
set -euo pipefail

# Step 1: train CIFAR-100 ResNet-18 baseline classifier in OpenOOD.
# Usage:
#   bash TP-OOD/step1_train_cifar100_resnet18.sh [SEED] [GPU_ID] [NUM_WORKERS] [SAVE_INTERVAL]
# Example:
#   bash TP-OOD/step1_train_cifar100_resnet18.sh 0 0 8 10

SEED="${1:-0}"
GPU_ID="${2:-0}"
NUM_WORKERS="${3:-8}"
SAVE_INTERVAL="${4:-10}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python main.py \
  --config configs/datasets/cifar100/cifar100.yml \
  configs/preprocessors/base_preprocessor.yml \
  configs/networks/resnet18_32x32.yml \
  configs/pipelines/train/baseline.yml \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --recorder.save_all_models True \
  --recorder.save_interval "${SAVE_INTERVAL}" \
  --merge_option merge

echo "Training finished."
echo "Checkpoint interval: every ${SAVE_INTERVAL} epochs"
echo "Expected checkpoint path:"
echo "  ${PROJECT_ROOT}/results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt"
