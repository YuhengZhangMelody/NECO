#!/usr/bin/env bash
set -euo pipefail

# Step 1 (optional): train 3 seeds for robust OOD averaging.
# Usage:
#   bash NECO/step1_train_cifar100_resnet18_3seeds.sh [GPU_ID] [NUM_WORKERS]

GPU_ID="${1:-0}"
NUM_WORKERS="${2:-8}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for SEED in 0 1 2; do
  echo "==== Train seed ${SEED} ===="
  bash "${SCRIPT_DIR}/step1_train_cifar100_resnet18.sh" "${SEED}" "${GPU_ID}" "${NUM_WORKERS}"
done
