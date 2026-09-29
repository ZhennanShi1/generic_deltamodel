#!/bin/bash
# A script to download and post-process GEFSv12 real-time forecasts for XCONUS from AWS (https://noaa-gefs-pds.s3.amazonaws.com/index.html)
# More documentation on dataset: https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-gefs-pds
# Originally developed by A. Wood under USACE and Reclamation sponsored projects at NCAR (circa 2018-2024), with support from USACE and Reclamation
# Refactored by J. Sturtevant into bash, and to update the source location 

# Check if start and end date arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <start_date YYYYMMDD> <end_date YYYYMMDD>"
    exit 1
fi

# Set the start and end dates from arguments
START_DATE=$1
END_DATE=$2

# Region settings for clipping
REGION="XCONUS"
SOUTH=20.0
NORTH=55.0
WEST=230.0  # Converted from -130 to 0-360 range
EAST=300.0  # Converted from -60 to 0-360 range

# Dataset (see documentation here: https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-gefs-pds)
DATASET="pgrb2s"
RESOLUTION="p25"

# Base output directory
BASE_OUTPUT_DIR="/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/$REGION/fcst"
mkdir -p $BASE_OUTPUT_DIR

# List of all ensemble members
MEMBERS=("c00" "p01" "p02" "p03" "p04" "p05" "p06" "p07" "p08" "p09" "p10" \
        "p11" "p12" "p13" "p14" "p15" "p16" "p17" "p18" "p19" "p20")

#MEMBERS=("c00" "p01") # for debugging

# Variables to retain, including Q2M (specific humidity at 2m)
VARS=("APCP_surface" "DLWRF_surface" "DSWRF_surface" "PRES_surface" \
      "RH_2maboveground" "TMP_2maboveground" "UGRD_10maboveground" "VGRD_10maboveground")

# Forecast initialization to process
INIT_HOUR="00"

# Lead times to download (in hours, from 0-240h)
START_LEAD=3
END_LEAD=240
#END_LEAD=6
TIMESTEP=3

# Loop through each date in the range
CURRENT_DATE=$START_DATE
while [[ "$CURRENT_DATE" -le "$END_DATE" ]]; do
    YEAR=${CURRENT_DATE:0:4}
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$YEAR/$CURRENT_DATE/"
    mkdir -p $OUTPUT_DIR

    # Clean up old files in output dir (from incomplete dopwnloads)
    rm -f $OUTPUT_DIR/*.f???
    rm -f $OUTPUT_DIR/*.f???.nc
    rm -f $OUTPUT_DIR/*tmp

    ### SKIP CONCATENATION OF ALL MEMBERS, DUE TO OCCASSIONAL ERRORS FROM INDIVID. ENS MEMBERS
    ## Check if FINAL_OUTPUT exists; if so, skip to the next date
    #FINAL_OUTPUT="$OUTPUT_DIR/gefs_v12_fcst.${CURRENT_DATE}_all_members.$REGION.nc"
    #if [[ -f $FINAL_OUTPUT ]]; then
    #    echo "[INFO] $FINAL_OUTPUT already exists. Skipping to the next date."
    #    CURRENT_DATE=$(date -d "$CURRENT_DATE +1 day" +%Y%m%d)
    #    continue
    #else
    #    echo "Processing $FINAL_OUTPUT for $CURRENT_DATE..."
    #fi

    # Loop through each ensemble member
    for MEMBER in "${MEMBERS[@]}"; do
        MEMBER_FILES=()
        #BASE_PATH="s3://noaa-gefs-pds/gefs.$CURRENT_DATE/$INIT_HOUR/$DATASET/"  # GEFSv11
        BASE_PATH="s3://noaa-gefs-pds/gefs.$CURRENT_DATE/$INIT_HOUR/atmos/${DATASET}${RESOLUTION}/"
        # Example filepath: https://noaa-gefs-pds.s3.amazonaws.com/gefs.20200923/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f207

        MEMBER_OUTPUT="$OUTPUT_DIR/gefs_v12_fcst.${CURRENT_DATE}${INIT_HOUR}.${MEMBER}.$REGION.nc"
        if [[ -f $MEMBER_OUTPUT ]]; then
            echo "[INFO] $MEMBER_OUTPUT already exists. Skipping to the next date."
            continue
        else
            echo "Processing $MEMBER_OUTPUT for $CURRENT_DATE..."
        fi

        # Loop through all forecast hours (0 to 240 in 3-hour steps)
        for HOUR in $(seq -f "%03g" $START_LEAD $TIMESTEP $END_LEAD); do
            FILE_NAME="ge${MEMBER}.t${INIT_HOUR}z.${DATASET}.0${RESOLUTION}.f${HOUR}"
            FULL_S3_PATH="${BASE_PATH}${FILE_NAME}"
            LOCAL_GRIB2="$OUTPUT_DIR/$FILE_NAME"
            LOCAL_NC="${LOCAL_GRIB2%.grb2}.nc"

            echo "Downloading $FULL_S3_PATH..."
            aws s3 cp "$FULL_S3_PATH" "$LOCAL_GRIB2" --no-sign-request

            if [[ -f $LOCAL_GRIB2 ]]; then
                echo "Converting $FILE_NAME to NetCDF format"
                wgrib2 $LOCAL_GRIB2 -netcdf $LOCAL_NC
                rm -rf $LOCAL_GRIB2

                echo "Clipping data for $FILE_NAME to the $REGION region"
                ncks -d latitude,$SOUTH,$NORTH -d longitude,$WEST,$EAST $LOCAL_NC ${LOCAL_NC%.nc}_clipped.nc
                mv ${LOCAL_NC%.nc}_clipped.nc $LOCAL_NC

                echo "Filtering selected variables for $LOCAL_NC"
                ncks -v $(IFS=","; echo "${VARS[*]}") $LOCAL_NC ${LOCAL_NC%.nc}_filtered.nc
                mv ${LOCAL_NC%.nc}_filtered.nc $LOCAL_NC

                # Compute specific humidity at 2m
                echo "Computing specific humidity (SPFH_2maboveground) for $LOCAL_NC"
                ncap2 -O -s "es=6.112*exp((17.67*(TMP_2maboveground-273.15))/(TMP_2maboveground-29.65)); \
                            SPFH_2maboveground=(RH_2maboveground/100.0) * es / (PRES_surface - 0.378*es)" $LOCAL_NC ${LOCAL_NC%.nc}_spfh2m.nc
                mv ${LOCAL_NC%.nc}_spfh2m.nc $LOCAL_NC

                # Add attributes to specific humidity
                ncatted -O -a units,SPFH_2maboveground,o,c,"kg/kg" \
                            -a long_name,SPFH_2maboveground,o,c,"Specific humidity at 2m" $LOCAL_NC

                # Remove extra variables
                echo "Removing RH_2maboveground and es from $LOCAL_NC"
                ncks -O -x -v RH_2maboveground,es $LOCAL_NC ${LOCAL_NC%.nc}_filtered.nc
                mv ${LOCAL_NC%.nc}_filtered.nc $LOCAL_NC

                MEMBER_FILES+=("$LOCAL_NC")
            else
                echo "Warning: $FULL_S3_PATH could not be downloaded. Skipping."
                continue
            fi
        done

        # Combine all forecast hours for this member into one file
        MEMBER_OUTPUT="$OUTPUT_DIR/gefs_v12_fcst.${CURRENT_DATE}${INIT_HOUR}.${MEMBER}.$REGION.nc"
        echo "Appending all forecast hours for member $MEMBER into $MEMBER_OUTPUT..."
        ncrcat -O -h "${MEMBER_FILES[@]}" $MEMBER_OUTPUT

        # Add ens_member dimension
        echo "Adding ens_member dimension for $MEMBER..."
        ncecat -O -u ens_member $MEMBER_OUTPUT ${MEMBER_OUTPUT%.nc}_ens.nc

        mv ${MEMBER_OUTPUT%.nc}_ens.nc $MEMBER_OUTPUT

        # Clean up individual forecast hour files
        echo "Cleaning up individual forecast hour files for $MEMBER"
        rm -f "${MEMBER_FILES[@]}"

    done

    ## Combine all members into one file
    #echo "Combining all members for $CURRENT_DATE into $FINAL_OUTPUT"
    #ncrcat -O -h $OUTPUT_DIR/gefs_v12_fcst.${CURRENT_DATE}.*.$REGION.nc $FINAL_OUTPUT

    # Clean up individual ensemble member files
    #echo "Cleaning up individual ensemble member files for $CURRENT_DATE"
    #rm -f $OUTPUT_DIR/gefs_v12_fcst.${CURRENT_DATE}.*.$REGION.nc

    CURRENT_DATE=$(date -d "$CURRENT_DATE +1 day" +%Y%m%d)

    echo " "; echo " "
done

echo "All operations completed successfully."