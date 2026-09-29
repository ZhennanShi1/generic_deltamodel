#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
###SBATCH --time=01:00:00
#SBATCH --output=./logs/get_gefs_fcsts.%A_%a.out  
#SBATCH --error=./logs/get_gefs_fcsts.%A_%a.err
#SBATCH --account=2306081036
#SBATCH --job-name=get_gefs_fcsts
#SBATCH --array=1-294  # Adjust this range as needed
###SBATCH --array=1  # Adjust this range as needed

# --- Define start and end dates ---
#START_DATE=20200101
#END_DATE=20230930
# START_DATE=20231001
# START_DATE=20231201
START_DATE=20231212
END_DATE=20240930
#END_DATE=20231206
# END_DATE=20241231
# NUM_JOBS=72   # This should match the array jobs above
NUM_JOBS=294   # This should match the array jobs above

# --- Make logdir ---
mkdir -p ./logs

# --- Load modules, set environment ---
echo "Loading modules..."
eval "$(conda shell.bash hook)"
conda activate npl-2024b
module load apps/nco/5.2.8
module load utility/wgrib2/3.0.2
# conda activate wgrib2_env

# Add wgrib2 to PATH
export PATH=/sw/utility/wgrib2/3.0.2/grib2/wgrib2:$PATH

# Convert start and end dates to epoch days
START_EPOCH=$(date -d "$START_DATE" +%s)
END_EPOCH=$(date -d "$END_DATE" +%s)

# Compute total days and days per job
TOTAL_DAYS=$(( (END_EPOCH - START_EPOCH) / 86400 + 1 ))
DAYS_PER_JOB=$(( TOTAL_DAYS / NUM_JOBS ))

# Ensure SLURM_ARRAY_TASK_ID is set (for manual testing)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "SLURM_ARRAY_TASK_ID is not set, defaulting to 1"
    SLURM_ARRAY_TASK_ID=1
fi

# Compute job-specific start and end dates
JOB_INDEX=$SLURM_ARRAY_TASK_ID
JOB_START_EPOCH=$(( START_EPOCH + (JOB_INDEX - 1) * DAYS_PER_JOB * 86400 ))
JOB_END_EPOCH=$(( JOB_START_EPOCH + (DAYS_PER_JOB - 1) * 86400 ))

# Convert back to YYYYMMDD format
JOB_START_DATE=$(date -d "@$JOB_START_EPOCH" +%Y%m%d)
JOB_END_DATE=$(date -d "@$JOB_END_EPOCH" +%Y%m%d)

# Ensure the last job runs to the final date
if [[ $JOB_INDEX -eq $NUM_JOBS ]]; then
    JOB_END_DATE=$END_DATE
fi

# Print debug info
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Running GEFS forecasts for job $JOB_INDEX from $JOB_START_DATE to $JOB_END_DATE"

# Run model with computed date range
sh ./get_gefsv12_fcst.sh $JOB_START_DATE $JOB_END_DATE
