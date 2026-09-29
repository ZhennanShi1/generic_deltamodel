#!/bin/bash
#SBATCH --job-name=process_GEFS_ensemble
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/zhennanshi/logs/process_ensemble_%A_%a.out
#SBATCH --error=/scratch/zhennanshi/logs/process_ensemble_%A_%a.err

echo "=========================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array Job ID  : $SLURM_ARRAY_JOB_ID"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Job Name      : $SLURM_JOB_NAME"
echo "Node          : $(hostname)"
echo "Start Time    : $(date)"
echo "=========================================="

source ~/.bashrc
conda activate dmg312

echo "Python: $(which python)"
python --version

ENSEMBLES=("ens01" "ens02" "ens03" "ens04" "ens05")
ENS=${ENSEMBLES[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing ensemble: $ENS"

python process_ensemble_files.py --ens $ENS

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="