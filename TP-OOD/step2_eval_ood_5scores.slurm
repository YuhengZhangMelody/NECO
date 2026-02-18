#!/bin/bash
#SBATCH --job-name=openood_c100_eval5
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs


source /home/ensta/ensta-yuheng.zhang/venvs/NECO/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 可通过 sbatch --export=ALL,ROOT=...,BATCH_SIZE=200,GPU_ID=0 覆盖
ROOT=${ROOT:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default}
BATCH_SIZE=${BATCH_SIZE:-200}
GPU_ID=${GPU_ID:-0}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

bash TP-OOD/step2_eval_ood_5scores.sh "${ROOT}" "${GPU_ID}" "${BATCH_SIZE}" \
  | tee "logs/eval5_cifar100_${SLURM_JOB_ID}.log"
