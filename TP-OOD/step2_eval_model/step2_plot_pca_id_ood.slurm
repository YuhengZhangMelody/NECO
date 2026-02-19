#!/bin/bash
#SBATCH --job-name=openood_c100_pca
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

source /home/ensta/ensta-yuheng.zhang/venvs/NECO/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 可通过 sbatch --export=ALL,CKPT_PATH=...,BATCH_SIZE=200,GPU_ID=0,ID_SPLIT=test,OOD_SPLIT=farood,OOD_DATASET=all,OUTPUT_DIR=... 覆盖
CKPT_PATH=${CKPT_PATH:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/model_epoch100.ckpt}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-200}
ID_SPLIT=${ID_SPLIT:-test}
OOD_SPLIT=${OOD_SPLIT:-farood}
OOD_DATASET=${OOD_DATASET:-all}
OUTPUT_DIR=${OUTPUT_DIR:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/pca}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

bash TP-OOD/step2_eval_model/step2_plot_pca_id_ood.sh \
  "${CKPT_PATH}" "${GPU_ID}" "${BATCH_SIZE}" "${ID_SPLIT}" "${OOD_SPLIT}" "${OOD_DATASET}" "${OUTPUT_DIR}" \
  | tee "logs/pca_id_ood_${SLURM_JOB_ID}.log"
