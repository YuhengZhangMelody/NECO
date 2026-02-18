#!/bin/bash
#SBATCH --job-name=openood_c100_ncplot
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# 你的 venv（按需修改）
source /home/ensta/ensta-yuheng.zhang/venvs/NECO/bin/activate

# 更稳定的 CPU 线程设置
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 可通过 sbatch --export=ALL,SEED_DIRS=s0,s1,s2,EPOCHS=10,20,30,40,50,60,70,80,90,100,DEVICE=cuda 覆盖
SEED_DIRS=${SEED_DIRS:-}
EPOCHS=${EPOCHS:-10,20,30,40,50,60,70,80,90,100}
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}
BATCH_SIZE=${BATCH_SIZE:-128}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

bash TP-OOD/step3_study_NC1-4/step3_plot_nc1-4.sh \
  "${SEED_DIRS}" "${EPOCHS}" "${DEVICE}" "${NUM_WORKERS}" "${BATCH_SIZE}" \
  | tee "logs/step3_plot_nc1-4_${SLURM_JOB_ID}.log"
