#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --time=24:00:00           # time in HH:MM:SS
#SBATCH --output=./logs/aorc_compress.%A_%a.out  # standard output with SLURM job and array task id
#SBATCH --error=./logs/aorc_compress.%A_%a.err   # standard error with SLURM job and array task id
#SBATCH --account=2306081036      # account number
#SBATCH --job-name=aorc_compress    # job name

# Ensure the logs directory exists
mkdir -p ./logs

# Create and activate Python virtual environment
module load apps/python3
conda activate npl-2024b

files_in=/beegfs/sets/aw-ciroh/common/lm_forcing/aorc/camels_ngen_v2_2/daily/aorc_daily.*.camels_ngen_hf.v2_2.nc
file_out=/beegfs/sets/aw-ciroh/common/lm_forcing/aorc/camels_ngen_v2_2/daily/aorc_daily.1979_2023.camels_ngen_hf.gz
echo "tar -cvzf ${file_out} ${files_in}"
tar -cvzf ${file_out} ${files_in}