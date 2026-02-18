#!/bin/bash
#SBATCH --job-name=openood_c100_nc5batch
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Optional overrides:
# sbatch --export=ALL,SEED_DIRS=s0,s1,s2,EPOCHS=10,20,...,DEVICE=cuda,NUM_WORKERS=8,BATCH_SIZE=128 run_nc5_batch_by_ood.slurm
SEED_DIRS=${SEED_DIRS:-s0,s1,s2}
EPOCHS=${EPOCHS:-10,20,30,40,50,60,70,80,90,100}
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}
BATCH_SIZE=${BATCH_SIZE:-128}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

bash TP-OOD/step4_study_NC5/run_nc5_batch_by_ood.sh \
  "${SEED_DIRS}" "${EPOCHS}" "${DEVICE}" "${NUM_WORKERS}" "${BATCH_SIZE}" \
  | tee "logs/run_nc5_batch_by_ood_${SLURM_JOB_ID}.log"
