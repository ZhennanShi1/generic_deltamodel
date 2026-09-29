#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1      # number of tasks per node
###SBATCH --mem=10G                 # memory allocation
#SBATCH --time=48:00:00          # walltime in HH:MM:SS
#SBATCH --output=/scratch/zhennanshi/logs/gefs_hcst2camels.%A_%a.out
#SBATCH --error=/scratch/zhennanshi/logs/gefs_hcst2camels.%A_%a.err
#SBATCH --account=2306081036         # account number
#SBATCH --job-name=gefs_hcst2camels  # job name
#SBATCH --array=1-20                 # array for years 2000-2019 # zhennan revised

# Ensure the logs directory exists
mkdir -p /scratch/zhennanshi/logs #zhennan revised

# Activate the Conda environment (if needed)
eval "$(conda shell.bash hook)"
conda activate dmg312 # zhennanrevised

# Calculate the year based on the SLURM array task ID
YEAR=$((1999 + SLURM_ARRAY_TASK_ID))  # For example, if SLURM_ARRAY_TASK_ID=1 then YEAR=2000

# Define start and end dates for the year in format YYYYMMDD
START_DATE="${YEAR}0101"  # January 1st
END_DATE="${YEAR}1231"    # zhennan temporarily set for testing
# START_DATE="${YEAR}1231"  # restart part way through year
# END_DATE="${YEAR}1231"    # December 31st

# Initialize current_date to the start date
current_date=$START_DATE

# Define script and mapping file paths
script="./apply_p2p.remap_ts.GEFS_ens.ngen.py"                   # p2p apply script for ngen inputs
# mapping_file="../mapping_gefs_xconus_to_camels_ngen_hf.v2_2.nc"  # pre-generated
mapping_file="../mapping_gefs_xconus_to_camels_gII.nc"  # pre-generated
time_shift="0h"                         # keep in UTC
hru_dtype="str"                         # data type for ngen-hf catchment (str)
append_ens_suffix="True"                # if True, script will label outputs with *ens01.nc, *ens02.nc etc.
# out_file_suffix="camels_ngen_hf_v2_2"   # do not include .nc
out_file_suffix="camels_gII"   # do not include .nc

# Loop over each day between START_DATE and END_DATE
while [[ "$current_date" -le "$END_DATE" ]]; do
    # Construct the input file name based on the current date (YYYYMMDD)
    # e.g., /beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/XCONUS/hcst/2000/20000101/gefs_v12_hcst.2000010100_all_members.XCONUS.nc
    in_file="/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/XCONUS/hcst/${YEAR}/gefs_v12_hcst.${current_date}00_all_members.XCONUS.nc"
    
    # If the input file exists, process it
    if [ -f "$in_file" ]; then
        # Construct output directory and output file name
        out_dir="/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/${out_file_suffix}/camels671/hcst/${YEAR}/${current_date}/"
        mkdir -p "$out_dir"
        file=$(basename "$in_file")
        
        # swap suffix
        out_file="$out_dir/${file/_all_members.XCONUS.nc/.$out_file_suffix.nc}"  
    
        echo "Processing file: $in_file"
        python "$script" "$mapping_file" "$in_file" "$out_file" "$hru_dtype" "$time_shift" "$append_ens_suffix"
        echo "Output file: $out_file"
    else
        echo "File not found: $in_file"
    fi
    
    # Increment current_date by one day.
    current_date=$(date -d "${current_date:0:4}-${current_date:4:2}-${current_date:6:2} +1 day" +%Y%m%d)
done
