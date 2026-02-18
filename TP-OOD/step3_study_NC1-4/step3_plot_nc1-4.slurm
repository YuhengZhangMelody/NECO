#!/bin/bash
#SBATCH --job-name=openood_nc1-5_plot
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

source /home/ensta/ensta-yuheng.zhang/venvs/NECO/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# sbatch --export=ALL,DATASETS=cifar100,cifar10,SEED_DIRS=s0,s1,s2,EPOCHS=20,40,60,80,100,OOD_DATASETS=all
DATASETS=${DATASETS:-cifar100}
SEED_DIRS=${SEED_DIRS:-}
EPOCHS=${EPOCHS:-10,20,30,40,50,60,70,80,90,100}
OOD_DATASETS=${OOD_DATASETS:-all}

cd '/home/ensta/ensta-yuheng.zhang/NECO'

bash TP-OOD/step3_study_NC1-4/step3_plot_nc1-4.sh \
  "${DATASETS}" "${SEED_DIRS}" "${EPOCHS}" "${OOD_DATASETS}" \
  | tee "logs/step3_plot_nc1-5_${SLURM_JOB_ID}.log"
