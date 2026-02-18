#!/bin/bash
#SBATCH --job-name=openood_c100_ncplot
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# 你的 venv（按需修改）
source /home/ensta/ensta-yuheng.zhang/venvs/NECO/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Optional overrides:
# sbatch --export=ALL,CSV_PATH=...,SEED_DIRS=s0,s1,s2,EPOCHS=10,20,...,OUTPUT_DIR=... step3_plot_nc1-4.slurm
CSV_PATH=${CSV_PATH:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc1-4_by_seed_epoch.csv}
SEED_DIRS=${SEED_DIRS:-}
EPOCHS=${EPOCHS:-10,20,30,40,50,60,70,80,90,100}
OUTPUT_DIR=${OUTPUT_DIR:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default/plots}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

bash TP-OOD/step3_study_NC1-4/step3_plot_nc1-4.sh \
  "${CSV_PATH}" "${SEED_DIRS}" "${EPOCHS}" "${OUTPUT_DIR}" \
  | tee "logs/step3_plot_nc1-4_${SLURM_JOB_ID}.log"
