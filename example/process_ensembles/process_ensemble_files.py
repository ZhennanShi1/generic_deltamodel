import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from pyet.radiation import hargreaves
import time

# Paths
GEFS_ROOT_DIR = '/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/camels_gII/hcst'
CAMELS_TOPO_FILE = '/beegfs/sets/aw-ciroh/common/datasets/camels/CAMELS_US/camels_attributes_v2.0/camels_topo.txt'
OUTPUT_BASE_DIR = '/u/st/dr/awwood/aw-ciroh-proj/projects/dl_da/daymet-gefs-camels-gII/'

# Load latitude for basins
topo_df = pd.read_csv(CAMELS_TOPO_FILE, sep=';')
topo_df['gauge_id'] = topo_df['gauge_id'].astype(str).str.zfill(8)
lat_dict = dict(zip(topo_df['gauge_id'], topo_df['gauge_lat']))

# Find all ensemble files
nc_files = glob.glob(os.path.join(GEFS_ROOT_DIR, '**', '*.nc'), recursive=True)
nc_files = [f for f in nc_files if 'ens' in f]

# Organize by ensemble
ensemble_files = {}
for f in nc_files:
    for ens in ['ens01', 'ens02', 'ens03', 'ens04', 'ens05']:
        if ens in f:
            ensemble_files.setdefault(ens, []).append(f)
            break

# Ensure consistent processing order
for ens_name in ['ens01', 'ens02', 'ens03', 'ens04', 'ens05']:
    if ens_name not in ensemble_files:
        continue

    files = sorted(ensemble_files[ens_name])
    print(f"\nProcessing ensemble: {ens_name} with {len(files)} files")
    output_dir = os.path.join(f"{OUTPUT_BASE_DIR}_{ens_name}")
    os.makedirs(output_dir, exist_ok=True)

    for nc_file in files:
        print(f"  Reading: {nc_file}")
        start_time = time.time()  # Start timer
        print("start at:", start_time)
        try:
            ds = xr.open_dataset(nc_file)
            catchment_ids = ds['catchment-id'].values
            catchment_ids = [cid.decode() if isinstance(cid, bytes) else cid for cid in catchment_ids]
            n_steps_per_day = 8
            n_days = ds.sizes['time'] // n_steps_per_day
            decoded_time = pd.to_datetime(ds['time'].values)
            daily_time = decoded_time[::n_steps_per_day]
        except Exception as e:
            print(f"  Skipping {nc_file}: {e}")
            continue

        for i, catch_id in enumerate(catchment_ids):
            lat_key = str(catch_id).zfill(8)
            lat = lat_dict.get(lat_key, np.nan)
            if np.isnan(lat):
                print(f"    Skipping {catch_id}: missing latitude.")
                continue

            try:
                prcp = ds['APCP_surface'].isel(**{'catchment-id': i}).values
                tmp = ds['TMP_2maboveground'].isel(**{'catchment-id': i}).values - 273.15
                srad = ds['DSWRF_surface'].isel(**{'catchment-id': i}).values
                pres = ds['PRES_surface'].isel(**{'catchment-id': i}).values
                q = ds['SPFH_2maboveground'].isel(**{'catchment-id': i}).values

                def daily(var): return var[:n_days * n_steps_per_day].reshape(n_days, n_steps_per_day)

                prcp_d = np.sum(daily(prcp), axis=1)
                tmax_d = np.max(daily(tmp), axis=1)
                tmin_d = np.min(daily(tmp), axis=1)
                tmean_d = np.mean(daily(tmp), axis=1)
                srad_d = np.mean(daily(srad), axis=1)
                q_d = np.mean(daily(q), axis=1)
                p_d = np.mean(daily(pres), axis=1)
                vp_d = q_d * p_d / (0.622 + 0.378 * q_d)

                pet_d = hargreaves(
                    tmean=pd.Series(tmean_d),
                    tmax=pd.Series(tmax_d),
                    tmin=pd.Series(tmin_d),
                    lat=np.radians(lat)
                ).values

                records = []
                for j in range(n_days):
                    records.append([
                        daily_time[j].year,
                        daily_time[j].month,
                        daily_time[j].day,
                        12,
                        43200,
                        prcp_d[j],
                        srad_d[j],
                        0,
                        tmax_d[j],
                        tmin_d[j],
                        vp_d[j],
                        pet_d[j]
                    ])

                out_df = pd.DataFrame(records, columns=[
                    'Year', 'Mnth', 'Day', 'Hr', 'dayl(s)',
                    'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)',
                    'tmin(C)', 'vp(Pa)', 'pet(mm/day)'
                ])
                out_path = os.path.join(output_dir, f"{lat_key}.txt")
                out_df.to_csv(out_path, sep=' ', index=False, float_format='%.3f', mode='a', header=not os.path.exists(out_path))

            except Exception as e:
                print(f"    Failed on basin {catch_id}: {e}")

