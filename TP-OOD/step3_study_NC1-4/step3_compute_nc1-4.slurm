#!/bin/bash
#SBATCH --job-name=openood_nc1-4_multi
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

source /home/ensta/ensta-yuheng.zhang/venvs/NECO/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# sbatch --export=ALL,SEED_DIRS=s0,s1,s2,EPOCHS=20,40,60,80,100,CHECKPOINTS_ROOT=results/cifar100_resnet18_32x32_base_e100_lr0.1_default
SEED_DIRS=${SEED_DIRS:-}
EPOCHS=${EPOCHS:-10,20,30,40,50,60,70,80,90,100}
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}
BATCH_SIZE=${BATCH_SIZE:-128}
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-results/cifar100_resnet18_32x32_base_e100_lr0.1_default}
OUTPUT_CSV=${OUTPUT_CSV:-${CHECKPOINTS_ROOT}/nc1-4_by_seed_epoch.csv}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

export CHECKPOINTS_ROOT
export OUTPUT_CSV

bash TP-OOD/step3_study_NC1-4/step3_compute_nc1-4.sh \
  "${SEED_DIRS}" "${EPOCHS}" "${DEVICE}" "${NUM_WORKERS}" "${BATCH_SIZE}" \
  | tee "logs/step3_nc1-4_${SLURM_JOB_ID}.log"
