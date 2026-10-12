
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# Configuration
# ============================================================

BASE_DIR = (
    "/beegfs/sets/aw-ciroh/common/met_fcsts/"
    "gefs_v12/camels_gII/camels671"
)

INIT_DATE = "2000010100"
ENS = "ens01"

ORIGINAL_DIR = os.path.join(BASE_DIR, "hcst")
PROCESSED_DIR = os.path.join(
    BASE_DIR, "zn_processed_hcst", f"gefs_processed_{ENS}"
)

ORIGINAL_NAME = (
    f"gefs_v12_hcst.{INIT_DATE}.camels_gII.{ENS}.nc"
)
PROCESSED_NAME = (
    f"gefs_processed.{INIT_DATE}.{ENS}.nc"
)

# ============================================================
# 1. Find original and processed files
# ============================================================

print("\n" + "=" * 65)
print("1. FIND FILES")
print("=" * 65)

original_matches = glob.glob(
    os.path.join(ORIGINAL_DIR, "**", ORIGINAL_NAME),
    recursive=True,
)

if not original_matches:
    raise FileNotFoundError(
        f"Original file not found: {ORIGINAL_NAME}"
    )

original_file = original_matches[0]
processed_file = os.path.join(
    PROCESSED_DIR, PROCESSED_NAME
)

if not os.path.isfile(processed_file):
    raise FileNotFoundError(processed_file)

print("Original:", original_file)
print("Processed:", processed_file)

# ============================================================
# 2. Open and compare datasets
# ============================================================

with xr.open_dataset(original_file) as original, \
     xr.open_dataset(processed_file) as processed:

    print("\n" + "=" * 65)
    print("2. DIMENSIONS")
    print("=" * 65)

    print("Original dimensions:", dict(original.sizes))
    print("Processed dimensions:", dict(processed.sizes))

    print("\n" + "=" * 65)
    print("3. VARIABLES")
    print("=" * 65)

    print("Original variables:")
    for var in original.data_vars:
        print(f"  {var}: {original[var].shape}")

    print("\nProcessed variables:")
    for var in processed.data_vars:
        print(f"  {var}: {processed[var].shape}")

    print("\n" + "=" * 65)
    print("4. INITIALIZATION AND FORECAST TIMES")
    print("=" * 65)

    original_times = pd.to_datetime(original["time"].values)
    processed_times = pd.to_datetime(processed["time"].values)

    print("Initialization from filename:", INIT_DATE)

    print("\nOriginal:")
    print("  Number of time steps:", len(original_times))
    print("  First timestamp:", original_times[0])
    print("  Last timestamp:", original_times[-1])
    print("  First 12 timestamps:")
    for t in original_times[:12]:
        print("   ", t)

    print("\n  Time interval counts:")
    print(pd.Series(original_times).diff().value_counts())

    print("\nProcessed:")
    print("  Number of forecast days:", len(processed_times))
    print("  All timestamps:")
    for i, t in enumerate(processed_times, start=1):
        print(f"    Day {i:02d}: {t}")

    print("\n" + "=" * 65)
    print("5. BASIN IDS")
    print("=" * 65)

    original_ids = original["catchment-id"].values
    processed_ids = processed["catchment-id"].values

    print("Original basin count:", len(original_ids))
    print("Processed basin count:", len(processed_ids))
    print("Original first 5 IDs:", original_ids[:5])
    print("Processed first 5 IDs:", processed_ids[:5])

    print(
        "Basin IDs identical:",
        np.array_equal(original_ids, processed_ids)
    )

    print("\n" + "=" * 65)
    print("6. CHECK DAILY AGGREGATION FOR BASIN 0")
    print("=" * 65)

    # Your processing script groups 8 three-hourly steps
    # into one forecast day.
    steps_per_day = 8
    n_days = len(original_times) // steps_per_day
    n_steps = n_days * steps_per_day

    if n_days != len(processed_times):
        print("WARNING: Different number of forecast days!")

    else:
        # Original APCP_surface -> processed prcp (daily sum)
        original_prcp = original["APCP_surface"].values[
            0, :n_steps
        ].reshape(n_days, steps_per_day).sum(axis=1)

        processed_prcp = processed["prcp"].values[0, :]

        # Original TMP_2maboveground -> processed tmean
        original_tmean = (
            original["TMP_2maboveground"].values[
                0, :n_steps
            ].reshape(n_days, steps_per_day).mean(axis=1)
            - 273.15
        )

        processed_tmean = processed["tmean"].values[0, :]

        comparison = pd.DataFrame({
            "date": processed_times,
            "original_prcp_aggregated": original_prcp,
            "processed_prcp": processed_prcp,
            "prcp_difference": (
                processed_prcp - original_prcp
            ),
            "original_tmean_aggregated": original_tmean,
            "processed_tmean": processed_tmean,
            "tmean_difference": (
                processed_tmean - original_tmean
            ),
        })

        print(comparison.to_string(index=False))

        print("\nPrecipitation matches:",
              np.allclose(
                  original_prcp, processed_prcp,
                  rtol=1e-5, atol=1e-5,
                  equal_nan=True
              ))

        print("Temperature matches:",
              np.allclose(
                  original_tmean, processed_tmean,
                  rtol=1e-5, atol=1e-5,
                  equal_nan=True
              ))

    print("\n" + "=" * 65)
    print("7. CHECK FORECAST TIME OVERLAP")
    print("=" * 65)

    next_init = (
        pd.Timestamp(INIT_DATE[:8])
        + pd.Timedelta(days=1)
    ).strftime("%Y%m%d") + INIT_DATE[8:10]

    next_file = os.path.join(
        PROCESSED_DIR,
        f"gefs_processed.{next_init}.{ENS}.nc"
    )

    if os.path.isfile(next_file):
        with xr.open_dataset(next_file) as next_ds:
            next_times = pd.to_datetime(
                next_ds["time"].values
            )

            overlap = np.intersect1d(
                processed_times.values,
                next_times.values
            )

            print("Next initialization:", next_init)
            print("Overlapping forecast dates:",
                  len(overlap))

            print("Shared dates:")
            for t in overlap:
                print(" ", pd.Timestamp(t))

            # Compare values for one common valid date.
            if len(overlap) > 0:
                common_date = pd.Timestamp(overlap[0])

                current_value = (
                    processed["prcp"]
                    .isel({"catchment-id": 0})
                    .sel(time=common_date)
                    .item()
                )

                next_value = (
                    next_ds["prcp"]
                    .isel({"catchment-id": 0})
                    .sel(time=common_date)
                    .item()
                )

                print("\nSame valid date:", common_date)
                print("Current init prcp:", current_value)
                print("Next init prcp:", next_value)
                print("Difference:", current_value - next_value)

    else:
        print("Next initialization file not found:", next_file)

print("\n" + "=" * 65)
print("CHECK COMPLETED")
print("=" * 65)
