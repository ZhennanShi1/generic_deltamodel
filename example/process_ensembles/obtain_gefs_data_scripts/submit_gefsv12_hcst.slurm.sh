#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/zhennanshi/logs/get_gefsv12.xconus.%A_%a.out
#SBATCH --error=/scratch/zhennanshi/logs/get_gefsv12.xconus.%A_%a.err
#SBATCH --account=2306081036
#SBATCH --job-name=get_gefs_c00.%A  
#SBATCH --array=1-20  # Adjust this range as needed for the number of years, zhennan first set it to 20

# Ensure the logs directory exists
mkdir -p /scratch/zhennanshi/logs

# --- Load modules, set environment ---
echo "Loading modules..."
eval "$(conda shell.bash hook)"
conda activate dmg312
module load apps/nco/5.2.8
module load utility/wgrib2/3.0.2

# Add wgrib2 to PATH
export PATH=/sw/utility/wgrib2/3.0.2/grib2/wgrib2:$PATH

# Define the start year and calculate the year for the current job array index
START_YEAR=2000
YEAR=$((START_YEAR + SLURM_ARRAY_TASK_ID - 1))

# Run the script for the calculated year and ensemble member
script="./get_gefsv12_hcst.sh"

echo "Downloading GEFSv12 hindcasts for year $YEAR across XCONUS"
sh $script $YEAR
