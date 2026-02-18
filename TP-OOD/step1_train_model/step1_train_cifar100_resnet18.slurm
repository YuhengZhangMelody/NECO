#!/bin/bash
#SBATCH --job-name=openood_c100_r18
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# 你的 venv（按需修改）
source /home/ensta/ensta-yuheng.zhang/venvs/NECO/bin/activate

# 更稳定的 CPU 线程设置
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 可通过 sbatch --export=ALL,SEED=0,NUM_WORKERS=8,SAVE_INTERVAL=10 提供；否则用默认值
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}
GPU_ID=${GPU_ID:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

bash TP-OOD/step1_train_model/step1_train_cifar100_resnet18.sh "${SEED}" "${GPU_ID}" "${NUM_WORKERS}" "${SAVE_INTERVAL}" \
  | tee "logs/train_cifar100_r18_s${SEED}_${SLURM_JOB_ID}.log"
