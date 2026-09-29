import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from pyet.radiation import hargreaves
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ens", required=True, help="Ensemble name, e.g., ens01")
args = parser.parse_args()

TARGET_ENS = args.ens

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
GEFS_ROOT_DIR = (
    '/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/'
    'camels_gII/camels671/hcst'
)

CAMELS_TOPO_FILE = (
    '/beegfs/scratch/projects/aw-ciroh-staged/common/datasets/'
    'camels/CAMELS_US/camels_attributes_v2.0/camels_topo.txt'
)

OUTPUT_BASE_DIR = (
    '/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/'
    'camels_gII/camels671/zn_processed_hcst'
)

# ---------------------------------------------------------
# Load CAMELS basin latitude
# ---------------------------------------------------------
topo_df = pd.read_csv(CAMELS_TOPO_FILE, sep=';')

topo_df['gauge_id'] = (
    topo_df['gauge_id']
    .astype(str)
    .str.zfill(8)
)

lat_dict = dict(
    zip(
        topo_df['gauge_id'],
        topo_df['gauge_lat']
    )
)

# ---------------------------------------------------------
# Find files for requested ensemble only
# ---------------------------------------------------------
ens_name = TARGET_ENS

files = sorted(
    glob.glob(
        os.path.join(
            GEFS_ROOT_DIR,
            '**',
            f'*.{ens_name}.nc'
        ),
        recursive=True
    )
)

if not files:
    print(f"No files found for {ens_name}")
    raise SystemExit(1)

print(
    f"\nProcessing ensemble: {ens_name} "
    f"with {len(files)} files"
)

# ---------------------------------------------------------
# Ensemble output directory
# ---------------------------------------------------------
output_dir = os.path.join(
    OUTPUT_BASE_DIR,
    f"gefs_processed_{ens_name}"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# ---------------------------------------------------------
# Process each initialization
# ---------------------------------------------------------
for nc_file in files:

    start_time = time.time()

    basename = os.path.basename(nc_file)

    # Example:
    # gefs_v12_hcst.2000010100.camels_gII.ens01.nc
    init_datetime = basename.split('.')[1]

    print("==========================================")
    print(f"Initialization: {init_datetime}")
    print(f"Reading       : {nc_file}")

    try:

        with xr.open_dataset(nc_file) as ds:

            ds.load()

            # ---------------------------------------------
            # Catchment IDs
            # ---------------------------------------------
            catchment_ids = ds['catchment-id'].values

            catchment_ids = np.array([
                cid.decode() if isinstance(cid, bytes)
                else str(cid)
                for cid in catchment_ids
            ])

            n_catchments = len(catchment_ids)

            # ---------------------------------------------
            # Basin latitude
            # ---------------------------------------------
            basin_lats = []

            for catch_id in catchment_ids:

                lat_key = (
                    str(catch_id)
                    .replace('cat-', '')
                    .zfill(8)
                )

                lat = lat_dict.get(
                    lat_key,
                    np.nan
                )

                basin_lats.append(lat)

            basin_lats = np.array(basin_lats)

            if np.isnan(basin_lats).any():

                missing = np.sum(
                    np.isnan(basin_lats)
                )

                print(
                    f"WARNING: {missing} basins "
                    f"have missing latitude"
                )

            # ---------------------------------------------
            # Time setup
            # ---------------------------------------------
            n_steps_per_day = 8

            n_days = (
                ds.sizes['time']
                // n_steps_per_day
            )

            n_used_steps = (
                n_days
                * n_steps_per_day
            )

            decoded_time = pd.to_datetime(
                ds['time'].values
            )

            daily_time = decoded_time[
                :n_used_steps:n_steps_per_day
            ]

            # ---------------------------------------------
            # Read arrays
            #
            # Shape:
            # (671, 80)
            # ---------------------------------------------
            prcp = (
                ds['APCP_surface']
                .values[:, :n_used_steps]
            )

            tmp = (
                ds['TMP_2maboveground']
                .values[:, :n_used_steps]
                - 273.15
            )

            srad = (
                ds['DSWRF_surface']
                .values[:, :n_used_steps]
            )

            pres = (
                ds['PRES_surface']
                .values[:, :n_used_steps]
            )

            q = (
                ds['SPFH_2maboveground']
                .values[:, :n_used_steps]
            )

            # ---------------------------------------------
            # Reshape:
            #
            # (671, 80)
            # ->
            # (671, 10, 8)
            # ---------------------------------------------
            def daily_reshape(var):

                return var.reshape(
                    n_catchments,
                    n_days,
                    n_steps_per_day
                )

            prcp_3h = daily_reshape(prcp)
            tmp_3h = daily_reshape(tmp)
            srad_3h = daily_reshape(srad)
            pres_3h = daily_reshape(pres)
            q_3h = daily_reshape(q)

            # ---------------------------------------------
            # Daily aggregation
            #
            # Result shape:
            # (671, 10)
            # ---------------------------------------------
            prcp_d = np.sum(
                prcp_3h,
                axis=2
            )

            tmax_d = np.max(
                tmp_3h,
                axis=2
            )

            tmin_d = np.min(
                tmp_3h,
                axis=2
            )

            tmean_d = np.mean(
                tmp_3h,
                axis=2
            )

            srad_d = np.mean(
                srad_3h,
                axis=2
            )

            q_d = np.mean(
                q_3h,
                axis=2
            )

            p_d = np.mean(
                pres_3h,
                axis=2
            )

            # ---------------------------------------------
            # Vapor pressure
            # ---------------------------------------------
            vp_d = (
                q_d * p_d
                /
                (
                    0.622
                    + 0.378 * q_d
                )
            )

            # ---------------------------------------------
            # PET
            #
            # pyet works basin-by-basin here because
            # latitude differs for each basin.
            #
            # Shape:
            # (671, 10)
            # ---------------------------------------------
            pet_d = np.full(
                (n_catchments, n_days),
                np.nan
            )

            for i in range(n_catchments):

                lat = basin_lats[i]

                if np.isnan(lat):
                    continue

                pet_d[i, :] = hargreaves(
                    tmean=pd.Series(
                        tmean_d[i, :]
                    ),
                    tmax=pd.Series(
                        tmax_d[i, :]
                    ),
                    tmin=pd.Series(
                        tmin_d[i, :]
                    ),
                    lat=np.radians(lat)
                ).values

            # ---------------------------------------------
            # Build processed xarray Dataset
            # ---------------------------------------------
            out_ds = xr.Dataset(

                data_vars={

                    'prcp': (
                        ('catchment-id', 'time'),
                        prcp_d
                    ),

                    'srad': (
                        ('catchment-id', 'time'),
                        srad_d
                    ),

                    'tmax': (
                        ('catchment-id', 'time'),
                        tmax_d
                    ),

                    'tmin': (
                        ('catchment-id', 'time'),
                        tmin_d
                    ),

                    'tmean': (
                        ('catchment-id', 'time'),
                        tmean_d
                    ),

                    'vp': (
                        ('catchment-id', 'time'),
                        vp_d
                    ),

                    'pet': (
                        ('catchment-id', 'time'),
                        pet_d
                    ),

                    'dayl': (
                        ('catchment-id', 'time'),
                        np.full(
                            (n_catchments, n_days),
                            43200.0
                        )
                    ),

                    'swe': (
                        ('catchment-id', 'time'),
                        np.zeros(
                            (n_catchments, n_days)
                        )
                    )
                },

                coords={

                    'catchment-id': catchment_ids,

                    'time': daily_time,

                    'latitude': (
                        'catchment-id',
                        basin_lats
                    )
                },

                attrs={
                    'initialization_time':
                        init_datetime,

                    'ensemble':
                        ens_name,

                    'description':
                        'Daily GEFSv12 forcing '
                        'remapped to 671 CAMELS basins'
                }
            )

            # ---------------------------------------------
            # Variable metadata
            # ---------------------------------------------
            out_ds['prcp'].attrs[
                'units'
            ] = 'mm/day'

            out_ds['srad'].attrs[
                'units'
            ] = 'W m-2'

            out_ds['tmax'].attrs[
                'units'
            ] = 'degC'

            out_ds['tmin'].attrs[
                'units'
            ] = 'degC'

            out_ds['tmean'].attrs[
                'units'
            ] = 'degC'

            out_ds['vp'].attrs[
                'units'
            ] = 'Pa'

            out_ds['pet'].attrs[
                'units'
            ] = 'mm/day'

            out_ds['dayl'].attrs[
                'units'
            ] = 's'

            out_ds['swe'].attrs[
                'units'
            ] = 'mm'

            # ---------------------------------------------
            # Output filename
            # ---------------------------------------------
            out_file = os.path.join(
                output_dir,
                (
                    f"gefs_processed."
                    f"{init_datetime}."
                    f"{ens_name}.nc"
                )
            )

            # ---------------------------------------------
            # Compression
            # ---------------------------------------------
            encoding = {}

            for var in out_ds.data_vars:

                encoding[var] = {
                    'zlib': True,
                    'complevel': 4
                }

            # ---------------------------------------------
            # Save NetCDF
            # ---------------------------------------------
            out_ds.to_netcdf(
                out_file,
                encoding=encoding
            )

            print(
                f"Output        : {out_file}"
            )

    except Exception as e:

        print(
            f"FAILED: {nc_file}"
        )

        print(
            f"Reason: {e}"
        )

        continue

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Finished {init_datetime} "
        f"in {elapsed:.1f} seconds"
    )