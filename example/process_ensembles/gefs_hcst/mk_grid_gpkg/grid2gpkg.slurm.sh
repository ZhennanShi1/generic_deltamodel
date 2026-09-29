#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1       # number of tasks per node
#SBATCH --mem=192G 
#SBATCH --time=06:00:00           # time in HH:MM:SS
#SBATCH --output=./logs/grid2gpkg.%j.out  # standard output with SLURM job id
#SBATCH --error=./logs/grid2gpkg.%j.err   # standard error with SLURM job id
#SBATCH --account=2306081036      # account number
#SBATCH --job-name=grid2gpkg    # job name

# Ensure the logs directory exists
mkdir -p ./logs

conda_init_script=/sw/apps/python3/anaconda-2023.07/etc/profile.d/conda.sh
. $conda_init_script
conda activate xesmf_env
module load apps/python3

# args
python grid2gpkg.py3.py aorc.v1_1.1km.grid.proj.tif aorc.v1_1.1km.grid.gpkg gpkg
