#!/usr/bin/env bash
set -euo pipefail

# Plot NC1-NC4 from precomputed CSV for one or multiple datasets.
#
# Usage:
#   bash TP-OOD/step3_study_NC1-4/step3_plot_nc1-4.sh \
#     [DATASETS] [SEED_DIRS] [EPOCHS] [OOD_SPLIT] [OOD_DATASETS]

DATASETS="${1:-cifar100}"
SEED_DIRS="${2:-}"
EPOCHS="${3:-10,20,30,40,50,60,70,80,90,100}"
OOD_SPLIT="${4:-farood}"
OOD_DATASETS="${5:-all}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CKPT_ROOT_TEMPLATE="${CKPT_ROOT_TEMPLATE:-results/{dataset}_resnet18_32x32_base_e100_lr0.1_default}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a DATASET_ARR <<< "${DATASETS}"

for dataset in "${DATASET_ARR[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [ -z "${dataset}" ] && continue

  ckpt_root="${CKPT_ROOT_TEMPLATE//\{dataset\}/${dataset}}"
  csv_path="${ckpt_root}/nc1-4_by_ood/nc1-4_${OOD_SPLIT}_${OOD_DATASETS}_by_seed_epoch.csv"
  output_dir="${ckpt_root}/nc1-4_by_ood/plots_${OOD_SPLIT}_${OOD_DATASETS}"

  echo "[plot] dataset=${dataset} csv=${csv_path}"

  python TP-OOD/step3_study_NC1-4/plot_nc1-4_diagnostics.py \
    --csv-path "${csv_path}" \
    --datasets "${dataset}" \
    --seed-dirs "${SEED_DIRS}" \
    --epochs "${EPOCHS}" \
    --ood-datasets "${OOD_DATASETS}" \
    --output-dir "${output_dir}"

  echo "[plot] saved under: ${PROJECT_ROOT}/${output_dir}"
done
