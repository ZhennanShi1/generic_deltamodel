The GEFS v12 ensemble meteorological forecasts (or hindcasts, for those with past initialization 
dates) were downloaded and processed with scripts developed at the National Center for 
Atmospheric Research (NCAR, Boulder CO), under projects sponsored by the US Army Corps of 
Engineers and US Bureau of Reclamation, led by PI Andy Wood (andywood@ucar.edu; awwood@mines.edu).

The dataset has been reproduced/processed here for use in projects led by Dr. Wood, sponsored by 
the NOAA Coop. Institute for R2O in Hydrology (CIROH).

If you are using these data in your project, please contact Dr. Wood for information on how to 
reference or acknowledge the source of the information.

AWW-20250228

Download and post-processing scripts (retrieval, NetCDF conversion, XCONUS clipping, SLURM
drivers) rewritten in bash by J. Sturtevant, 2024-2025, from earlier tools by A. Wood.


Notes:
 * GEFSv12 hindcast/reforecasts (2000-2019): https://noaa-gefs-retrospective.s3.amazonaws.com/index.html#GEFSv12/reforecast/
 * GEFSv11/v12 operational forecast archives (GEFSv12: 20200923 to present): https://noaa-gefs-pds.s3.amazonaws.com/index.html
 * Ensemble members:
   - hindcast  (get_gefsv12_hcst.sh):  5 members - c00 (control) + p01-p04
   - forecast  (get_gefsv12_fcst.sh): 21 members - c00 (control) + p01-p20
 * Lead times: 3 to 240 h at 3-hourly step (80 steps, f003-f240) for both datasets.
   - hindcast is pulled from the archive's 3-hourly "Days:1-10" files
   - the coarser 6-hourly "Days:10-16" reforecast files are not downloaded

The data organization would look like, 2000-2019:
/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/XCONUS/hcst/2000/
├── 20000101/
│   └── gefs_v12_hcst.2000010100_all_members.XCONUS.nc
├── 20000102/
│   └── gefs_v12_hcst.2000010200_all_members.XCONUS.nc
├── 20000103/
│   └── gefs_v12_hcst.2000010300_all_members.XCONUS.nc
...
