#!/usr/bin/env bash
set -euo pipefail

# Plot NC1-NC4 from precomputed CSV (no checkpoint re-evaluation).
#
# Usage:
#   bash TP-OOD/step3_study_NC1-4/step3_plot_nc1-4.sh \
#     [CSV_PATH] [SEED_DIRS] [EPOCHS] [OUTPUT_DIR]

CSV_PATH="${1:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc1-4_by_seed_epoch.csv}"
SEED_DIRS="${2:-}"
EPOCHS="${3:-10,20,30,40,50,60,70,80,90,100}"
OUTPUT_DIR="${4:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/plots}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[plot] csv=${CSV_PATH}"

python TP-OOD/step3_study_NC1-4/plot_nc1-4_diagnostics.py \
  --csv-path "${CSV_PATH}" \
  --seed-dirs "${SEED_DIRS}" \
  --epochs "${EPOCHS}" \
  --output-dir "${OUTPUT_DIR}"

echo "[plot] saved under: ${PROJECT_ROOT}/${OUTPUT_DIR}"
