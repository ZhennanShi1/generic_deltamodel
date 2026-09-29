#!/bin/bash
# A script to download and post-process GEFSv12 reforecasts from 2000-2019 for the XCONUS
# J. Sturtevant, Nov 2024

# Check if year argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <year>"
    exit 1
fi

# Set the year from the argument
YEAR=$1

# List of all ensemble members
MEMBERS=("c00" "p01" "p02" "p03" "p04")

# Region settings for clipping
REGION="XCONUS"
SOUTH=20.0
NORTH=55.0
WEST=230.0  # Converted from -130 to 0-360 range
EAST=300.0  # Converted from -60 to 0-360 range

# List of variables to download
VARS=("pres_sfc" "tmp_2m" "spfh_2m" "apcp_sfc" "dswrf_sfc" "dlwrf_sfc" "ugrd_hgt" "vgrd_hgt")

# Base output directory
OUTPUT_DIR_ROOT="/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/$REGION/hcst/$YEAR/"

# Create the flist directory if it doesn't exist
FLIST_DIR="./flists/"
mkdir -p $FLIST_DIR

# Check if the year is a leap year
if (( (YEAR % 4 == 0 && YEAR % 100 != 0) || (YEAR % 400 == 0) )); then
    DAYS_IN_YEAR=366
else
    DAYS_IN_YEAR=365
fi

# Loop through each day of the year
for DAY in $(seq -w 0 $((DAYS_IN_YEAR - 1))); do
    DATE=$(date -d "$YEAR-01-01 +$DAY days" +%Y%m%d00)
    DATE_DIR=$(echo $DATE | cut -c1-8)  # e.g., 20000101 from 2000010100
    OUTPUT_DIR="$OUTPUT_DIR_ROOT/$DATE_DIR/"
    mkdir -p $OUTPUT_DIR

    # Check if FINAL_OUTPUT exists; if so, skip to the next date
    FINAL_OUTPUT="$OUTPUT_DIR/gefs_v12_hcst.${DATE}_all_members.$REGION.nc"
    if [[ -f $FINAL_OUTPUT ]]; then
        echo "FINAL_OUTPUT for $DATE already exists. Skipping to the next date."
        continue
    fi
    
    # Loop through each ensemble member
    for MEMBER in "${MEMBERS[@]}"; do
        MEMBER_FILES=()

        # Base path for GEFSv12 reforecast data for the current member
        BASE_PATH="s3://noaa-gefs-retrospective/GEFSv12/reforecast/$YEAR/$DATE/$MEMBER/"
        
        # List all files in the specified directory on S3
        echo "Listing files on S3 for member $MEMBER on $DATE..."
        file_list=$FLIST_DIR/file_list_$MEMBER.$DATE.txt
        aws s3 ls $BASE_PATH --recursive --no-sign-request > $file_list

        # Loop to find and download each file that matches the variables in the VARS array and is in Days:1-10
        for VAR in "${VARS[@]}"; do
            grep "Days:1-10" $file_list | grep "$VAR" | grep -v '\.idx$' | while read -r line; do
                FILE_PATH=$(echo $line | awk '{print $4}')
                FULL_S3_PATH="s3://noaa-gefs-retrospective/$FILE_PATH"
                LOCAL_GRIB2="$OUTPUT_DIR/$(basename $FILE_PATH)"
                LOCAL_NC="${LOCAL_GRIB2%.grib2}.nc"
                
                echo "Downloading $FILE_PATH for member $MEMBER on $DATE..."
                aws s3 cp "$FULL_S3_PATH" "$LOCAL_GRIB2" --no-sign-request
                
                if [[ -f $LOCAL_GRIB2 ]]; then
                    echo "Converting $VAR to NetCDF format for member $MEMBER on $DATE..."
                    wgrib2 $LOCAL_GRIB2 -netcdf $LOCAL_NC
                    rm -rf $LOCAL_GRIB2
                    
                    # Clip the data to the specified lat/lon range
                    echo "Clipping data for $VAR to the $REGION region for member $MEMBER on $DATE..."
                    ncks -d latitude,$SOUTH,$NORTH -d longitude,$WEST,$EAST $LOCAL_NC ${LOCAL_NC%.nc}_clipped.nc
                    mv ${LOCAL_NC%.nc}_clipped.nc $LOCAL_NC
                    
                    # Add this file to the array for member file appending
                    MEMBER_FILES+=("$LOCAL_NC")
                    
                    # Keep only the 10m above ground variables for ugrd_hgt and vgrd_hgt
                    if [[ $VAR == "ugrd_hgt" ]]; then
                        echo "Filtering UGRD_10maboveground for member $MEMBER on $DATE..."
                        ncks -v UGRD_10maboveground $LOCAL_NC ${LOCAL_NC%.nc}_filtered.nc
                        mv ${LOCAL_NC%.nc}_filtered.nc $LOCAL_NC
                    elif [[ $VAR == "vgrd_hgt" ]]; then
                        echo "Filtering VGRD_10maboveground for member $MEMBER on $DATE..."
                        ncks -v VGRD_10maboveground $LOCAL_NC ${LOCAL_NC%.nc}_filtered.nc
                        mv ${LOCAL_NC%.nc}_filtered.nc $LOCAL_NC
                    fi
                
                else
                    echo "Error: Download of $VAR for member $MEMBER on $DATE failed."
                    exit 1
                fi
                    
            done
        done
        
        # Create a single NetCDF file for the member by appending variables
        MEMBER_OUTPUT="$OUTPUT_DIR/gefs_v12_hcst.${DATE}.${MEMBER}.$REGION.nc"
        echo "Appending all variables for member $MEMBER on $DATE into $MEMBER_OUTPUT..."
        FIRST_FILE=$(ls $OUTPUT_DIR/*${DATE}_${MEMBER}*.nc | head -n 1)
        mv $FIRST_FILE $MEMBER_OUTPUT

        for FILE in $OUTPUT_DIR/*${DATE}_${MEMBER}*.nc; do
            if [[ $FILE != $FIRST_FILE ]]; then
                ncks -A $FILE $MEMBER_OUTPUT
            fi
        done
        
        # Add ens_member dimension
        echo "Adding ens_member dimension for member $MEMBER..."
        ncecat -O -u ens_member $MEMBER_OUTPUT ${MEMBER_OUTPUT%.nc}_ens.nc

        # Add init_time as an unlimited dimension, then assign DATE to it
        echo "Adding init_time dimension with value $DATE for file $MEMBER_OUTPUT..."
        ncap2 -O -s 'defdim("init_time",1); init_time[init_time]=0' -s "init_time=${DATE}" ${MEMBER_OUTPUT%.nc}_ens.nc ${MEMBER_OUTPUT%.nc}_ens_time.nc

        # Move the final file back to the original name
        mv ${MEMBER_OUTPUT%.nc}_ens_time.nc $MEMBER_OUTPUT

        # Remove individual variable NetCDF files for the current ensemble member, but keep the final concatenated file
        echo "Cleaning up individual variable NetCDF files for member $MEMBER on $DATE..."
        rm $OUTPUT_DIR/*${DATE}_${MEMBER}*.nc
        rm ${MEMBER_OUTPUT%.nc}_ens.nc
    
    done
    # Combine all ensemble member files for the current day into one file
    echo "Combining all ensemble member files for $DATE into $FINAL_OUTPUT..."
    ncrcat -h $OUTPUT_DIR/gefs_v12_hcst.${DATE}.*.$REGION.nc $FINAL_OUTPUT

    # Clean up individual member NetCDF files
    echo "Cleaning up individual member NetCDF files for $DATE..."
    rm $OUTPUT_DIR/gefs_v12_hcst.${DATE}.*.$REGION.nc
done

echo "All operations completed successfully."