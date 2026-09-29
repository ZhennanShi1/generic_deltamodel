If you do not see the hcst data at /beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/XCONUS/hcst, you may want to download it first: 
    use the scripts under /u/au/ac/zhennanshi/znprojects/generic_deltamodel/example/process_ensembles/obtain_gefs_data_scripts

otherwise:   
    Remapping steps to 671 basins:
        step (1) re-calculating the spatial weights for the grid-to-polygon mapping, using /beegfs/sets/aw-ciroh/projects/tools/poly2poly/gefs_hcst/submit.sp_wts.sh  

        step (2) applying those weights in the remapping step, which is all set up to run as an array job parallelized by year, using /beegfs/sets/aw-ciroh/projects/tools/poly2poly/gefs_hcst/apply_p2p/submit.apply_gefs_hcst_p2p.slurm.sh

    Then you can run the submit_process_GEFS_data, the data structure saved to '/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/camels_gII/camels671/zn_processed_hcst', and it looks like this:
        zn_processed_hcst/
        ├── gefs_processed_ens01/
        │   ├── gefs_processed.2000010100.ens01.nc
        │   ├── gefs_processed.2000010200.ens01.nc
        │   ├── gefs_processed.2000010300.ens01.nc
        │   ├── ...
        │   └── gefs_processed.2019123000.ens01.nc
        │
        ├── gefs_processed_ens02/
        │   ├── gefs_processed.2000010100.ens02.nc
        │   ├── gefs_processed.2000010200.ens02.nc
        │   └── ...
        │
        ├── gefs_processed_ens03/
        │   └── ...
        │
        ├── gefs_processed_ens04/
        │   └── ...
        │
        └── gefs_processed_ens05/
            └── ...


    Each one of these files, eg. gefs_processed.2000010100.ens01.nc, contains all 671 basins and all 10 forecast days together:

        dimensions:
            catchment-id = 671
            time = 10

        and with variables:
        prcp    (671, 10)
        srad    (671, 10)
        tmax    (671, 10)
        tmin    (671, 10)
        tmean   (671, 10)
        vp      (671, 10)
        pet     (671, 10)
        dayl    (671, 10)
        swe     (671, 10)

    The overlap is still preserved. For example:

        gefs_processed.2000010100.ens01.nc
            Jan 1 → Jan 10

        gefs_processed.2000010200.ens01.nc
            Jan 2 → Jan 11

        gefs_processed.2000010300.ens01.nc
            Jan 3 → Jan 12

    2000010300 is a date + initialization hour, not just a date
        2000010300
        │   │ │ └─ 00 = hour (00 UTC)
        │   │ └─── 03 = day
        │   └───── 01 = month
        └───────── 2000 = year