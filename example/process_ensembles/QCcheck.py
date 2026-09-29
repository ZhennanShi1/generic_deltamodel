import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Files
# ---------------------------------------------------------
nc_file = (
    "/beegfs/sets/aw-ciroh/common/met_fcsts/gefs_v12/"
    "camels_gII/camels671/zn_processed_hcst/"
    "gefs_processed_ens01/"
    "gefs_processed.2008091700.ens01.nc" # RANDOMLY SELECT A DATE TO PLOT
)

topo_file = (
    "/beegfs/scratch/projects/aw-ciroh-staged/common/datasets/"
    "camels/CAMELS_US/camels_attributes_v2.0/"
    "camels_topo.txt"
)

# ---------------------------------------------------------
# Read processed GEFS
# ---------------------------------------------------------
ds = xr.open_dataset(nc_file)

# Mean temperature across the 10 forecast days
mean_temp = ds["tmean"].mean(dim="time").values

catch_ids = [
    str(x).replace("cat-", "").zfill(8)
    for x in ds["catchment-id"].values
]

# ---------------------------------------------------------
# Read CAMELS basin coordinates
# ---------------------------------------------------------
topo = pd.read_csv(topo_file, sep=";")

topo["gauge_id"] = (
    topo["gauge_id"]
    .astype(str)
    .str.zfill(8)
)

topo = topo.set_index("gauge_id")

lat = np.array([
    topo.loc[cid, "gauge_lat"]
    for cid in catch_ids
])

lon = np.array([
    topo.loc[cid, "gauge_lon"]
    for cid in catch_ids
])

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

sc = plt.scatter(
    lon,
    lat,
    c=mean_temp,
    s=25
)

plt.colorbar(
    sc,
    label="Mean air temperature (°C)"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "GEFS 671 Basins Remapping Check\n"
    "Mean Temperature, Init 2008-09-17, ens01"
)

plt.grid(alpha=0.2)

plt.tight_layout()

outfile = "gefs_camels_temperature_check.png"
plt.savefig(outfile, dpi=200)

print(f"Saved: {outfile}")
