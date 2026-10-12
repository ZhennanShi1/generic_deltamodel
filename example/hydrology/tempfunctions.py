import numpy as np
import pandas as pd
import json
import torch 
import os 
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.dmg.core.utils import Dates  


def get_config_path(model_type="d3"):
    """
    Return config path based on DHBV setup.

    model_type options:
        - "d3"     : DHBV 1.1p with 3 dynamic parameters
        - "d2"     : DHBV 1.1p with 2 dynamic parameters
        - "static" : DHBV 1.1p with no dynamic parameters (pure static parameters)
        - "1.0_d2" : DHBV 1.0 with 2 dynamic parameters ###waiting to be added
    """
    config_map = {
        "d3": "../example/conf/config_dhbv_1_1p_3dyn.yaml",
        "d2": "../example/conf/config_dhbv_1_1p_2dyn.yaml",
        "static": "../example/conf/config_dhbv_1_1p_0dyn.yaml", 
    }

    if model_type not in config_map:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from {list(config_map.keys())}"
        )

    config_path = config_map[model_type]

    return config_path
 
def nse(sim, obs):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - obs.mean())**2)

def kge(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta  = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Eq. (5): Empirical CDF F(y)
# ============================================================

def empirical_cdf(esp, y):
    """
    esp : shape (n_members,)
        ESP ensemble streamflow forecasts at one time.
    y : scalar
        Possible streamflow value.
    """
    esp = np.asarray(esp)
    return np.mean(esp <= y)


# ============================================================
# Eq. (6): Observation step function F0(y)
# ============================================================

def observation_cdf(obs, y):
    """
    F0(y) =
        0, y < obs
        1, y >= obs
    """
    return 1.0 if y >= obs else 0.0


# ============================================================
# Eq. (5): CRPS for one forecast case
# ============================================================

def crps_single(esp, obs, num_grid=1000):
    """
    esp : shape (n_members,)
        Ensemble forecast for one basin, one init date, one lead time.
    obs : scalar
        Observed/reference streamflow at target date.
    """

    esp = np.asarray(esp, dtype=float)

    if np.all(np.isnan(esp)) or np.isnan(obs):
        return np.nan

    y_min = min(np.nanmin(esp), obs)
    y_max = max(np.nanmax(esp), obs)

    if y_min == y_max:
        return 0.0

    ys = np.linspace(y_min, y_max, num_grid)
    dy = ys[1] - ys[0]

    integral = 0.0

    for y in ys:
        F_y = empirical_cdf(esp, y)
        F0_y = observation_cdf(obs, y)
        integral += (F_y - F0_y) ** 2

    return integral * dy


# ============================================================
# Average CRPS_forecast for one basin and one lead time
# ============================================================

def average_crps(esps, obs):
    """
    esps : shape (n_members, n_init_dates)
        Ensemble forecasts for one basin and one lead time.
    obs : shape (n_init_dates,)
        Reference streamflow at target dates.
    """

    esps = np.asarray(esps, dtype=float)
    obs = np.asarray(obs, dtype=float)

    crps_values = []

    for t in range(len(obs)):
        esp_t = esps[:, t]
        obs_t = obs[t]

        crps_values.append(
            crps_single(esp_t, obs_t)
        )

    return np.nanmean(crps_values)


# ============================================================
# Eq. (7): CRPS_ref,ad
# climatological average discharge reference
# ============================================================

def crps_ref_climatology(qsim):
    """
    qsim : shape (n_total_days,)
        Continuous reference streamflow for one basin.

    CRPS_ref,ad =
        mean |qsim(t) - mean(qsim)|
    """

    qsim = np.asarray(qsim, dtype=float)
    q_mean = np.nanmean(qsim)

    return np.nanmean(np.abs(qsim - q_mean))


# ============================================================
# Eq. (8): CRPS_ref,pf(LT)
# persistence reference
# ============================================================

def crps_ref_persistence(qsim, init_indices, lead_time):
    """
    qsim : shape (n_total_days,)
        Continuous reference streamflow for one basin.

    init_indices : shape (n_init_dates,)
        Indices of forecast initialization dates.

    lead_time : int
        LT = 1, 2, ..., 15

    Persistence forecast:
        qsim(init_date)

    Target:
        qsim(init_date + LT)
    """

    qsim = np.asarray(qsim, dtype=float)

    target_indices = init_indices + lead_time
    init_indices = np.asarray(init_indices)

    target_q = qsim[target_indices]
    init_q = qsim[init_indices]

    return np.nanmean(np.abs(target_q - init_q))


# ============================================================
# Eq. (4): CRPSS for one basin and one lead time
# ============================================================

def crpss_one_basin_one_lead(
    esps,
    obs,
    qsim,
    init_indices,
    lead_time,
    reference_type="persistence"
):
    """
    esps : shape (n_members, n_init_dates)
    obs : shape (n_init_dates,)
    qsim : shape (n_total_days,)
    init_indices : shape (n_init_dates,)
    lead_time : int
    """

    crps_forecast = average_crps(esps, obs)

    if reference_type == "climatology":
        crps_reference = crps_ref_climatology(qsim)

    elif reference_type == "persistence":
        crps_reference = crps_ref_persistence(
            qsim=qsim,
            init_indices=init_indices,
            lead_time=lead_time
        )

    else:
        raise ValueError("reference_type must be 'climatology' or 'persistence'.")

    crpss_value = (
        crps_reference - crps_forecast
    ) / crps_reference

    return crpss_value, crps_forecast, crps_reference


# ============================================================
# Main function: compute CRPSS for all basins and lead times
# ============================================================

def compute_crpss_all_basins(
    forecasts,
    qsim,
    init_indices,
    max_lead=15
):
    """
    forecasts : shape (n_basins, n_init_dates, n_lead_times, n_members)
    qsim : shape (n_basins, n_total_days)
    init_indices : shape (n_init_dates,)
    max_lead : int
    """

    forecasts = np.asarray(forecasts, dtype=float)
    qsim = np.asarray(qsim, dtype=float)
    init_indices = np.asarray(init_indices)

    n_basins = forecasts.shape[0]

    crpss_pf_all = np.full((n_basins, max_lead), np.nan)
    crpss_ad_all = np.full((n_basins, max_lead), np.nan)

    crps_forecast_all = np.full((n_basins, max_lead), np.nan)
    crps_ref_pf_all = np.full((n_basins, max_lead), np.nan)
    crps_ref_ad_all = np.full((n_basins, max_lead), np.nan)

    for b in range(n_basins):

        qsim_basin = qsim[b, :]

        for lt in range(1, max_lead + 1):

            # forecasts for one basin and one lead time
            # original shape: (n_init_dates, n_members)
            esps_lt = forecasts[b, :, lt - 1, :]

            # convert to shape required by average_crps:
            # (n_members, n_init_dates)
            esps_lt = esps_lt.T

            # target dates = init date + lead time
            target_indices = init_indices + lt
            obs_lt = qsim_basin[target_indices]

            # CRPSS using persistence reference
            crpss_pf, crps_fc, crps_ref_pf = crpss_one_basin_one_lead(
                esps=esps_lt,
                obs=obs_lt,
                qsim=qsim_basin,
                init_indices=init_indices,
                lead_time=lt,
                reference_type="persistence"
            )

            # CRPSS using climatological average discharge reference
            crpss_ad, _, crps_ref_ad = crpss_one_basin_one_lead(
                esps=esps_lt,
                obs=obs_lt,
                qsim=qsim_basin,
                init_indices=init_indices,
                lead_time=lt,
                reference_type="climatology"
            )

            crpss_pf_all[b, lt - 1] = crpss_pf
            crpss_ad_all[b, lt - 1] = crpss_ad

            crps_forecast_all[b, lt - 1] = crps_fc
            crps_ref_pf_all[b, lt - 1] = crps_ref_pf
            crps_ref_ad_all[b, lt - 1] = crps_ref_ad

    return {
        "CRPSS_pf": crpss_pf_all,
        "CRPSS_ad": crpss_ad_all,
        "CRPS_forecast": crps_forecast_all,
        "CRPS_ref_pf": crps_ref_pf_all,
        "CRPS_ref_ad": crps_ref_ad_all,
        "lead_times": np.arange(1, max_lead + 1)
    }


# ============================================================
# Plot paper-style figure
# ============================================================

def plot_crpss_lead_time(result):
    """
    result : output from compute_crpss_all_basins()
    """

    lead_times = result["lead_times"]

    crpss_pf_all = result["CRPSS_pf"]
    crpss_ad_all = result["CRPSS_ad"]

    # CRPSS_pf statistics across basins
    pf_mean = np.nanmean(crpss_pf_all, axis=0)
    pf_p05 = np.nanpercentile(crpss_pf_all, 5, axis=0)
    pf_p25 = np.nanpercentile(crpss_pf_all, 25, axis=0)
    pf_p75 = np.nanpercentile(crpss_pf_all, 75, axis=0)
    pf_p95 = np.nanpercentile(crpss_pf_all, 95, axis=0)

    # CRPSS_ad statistics across basins
    ad_mean = np.nanmean(crpss_ad_all, axis=0)
    ad_p25 = np.nanpercentile(crpss_ad_all, 25, axis=0)
    ad_p75 = np.nanpercentile(crpss_ad_all, 75, axis=0)

    plt.figure(figsize=(7, 5))

    # light gray: 5–95% range for CRPSS_pf
    plt.fill_between(
        lead_times,
        pf_p05,
        pf_p95,
        color="lightgray",
        label="CRPSS_pf 5–95%"
    )

    # dark gray: 25–75% range for CRPSS_pf
    plt.fill_between(
        lead_times,
        pf_p25,
        pf_p75,
        color="gray",
        alpha=0.6,
        label="CRPSS_pf 25–75%"
    )

    # solid black: mean CRPSS_pf
    plt.plot(
        lead_times,
        pf_mean,
        color="black",
        linewidth=2,
        label="CRPSS_pf"
    )

    # dashed black: mean CRPSS_ad
    plt.plot(
        lead_times,
        ad_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        label="CRPSS_ad"
    )

    # dotted black: 25–75% range for CRPSS_ad
    plt.plot(
        lead_times,
        ad_p25,
        color="black",
        linestyle=":",
        linewidth=1
    )

    plt.plot(
        lead_times,
        ad_p75,
        color="black",
        linestyle=":",
        linewidth=1
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.xlabel("Lead time [days]")
    plt.ylabel("CRPSS")
    plt.xlim(0, lead_times[-1] + 0.5)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def _clim_doy(dates):
    """
    Use a consistent 365-day climatological calendar.
    Feb 29 is assigned to Feb 28's DOY.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    doy = dates.dayofyear.to_numpy().copy()
    leap = dates.is_leap_year
    after_feb = dates.month > 2
    doy[leap & after_feb] -= 1
    return doy


def _season_mask(doys, target_doy, window):
    """Circular +/- window around a climatological DOY."""
    dist = np.abs(np.asarray(doys) - target_doy)
    return np.minimum(dist, 365 - dist) <= window

 


def compute_bias_correction_from_dataset(
    daymet_tensor,
    GEFS_DIR,
    timesteps,
    basin_idx,
    gage_id,
    ens_id,
    start_date,
    window=30,
    method="scalar",
):
    """
    Calculate GEFS bias corrections using only the previous
    `window` days before start_date.

    GEFS data:
        Separate NetCDF files for each initialization.
        Each file contains 10 forecast leads.

    Daymet data:
        Historical observed forcings from x_phy.

    Returns
    -------
    corrections[lead][var]

    Scalar:
        prcp, pet -> multiplicative ratio
        tmean     -> additive difference

    CDF:
        Quantile mapping for each variable and lead.
    """

    if method not in ("scalar", "cdf"):
        raise ValueError(f"Unknown correction method: {method}")

    if window < 1:
        raise ValueError("window must be at least 1 day")

    variables = ["prcp", "tmean", "pet"]

    pred_date = pd.Timestamp(start_date).normalize()
    hist_start = pred_date - pd.Timedelta(days=window)

    basin_id = str(gage_id).zfill(8)
    ens_name = f"ens{ens_id + 1:02d}"

    # --------------------------------------------------
    # 1. Historical Daymet data
    # --------------------------------------------------
    daymet_values = (
        daymet_tensor[:, basin_idx, :3]
        .detach()
        .cpu()
        .numpy()
    )

    df_daymet = pd.DataFrame(
        daymet_values,
        columns=variables,
    )

    df_daymet["date"] = pd.to_datetime(timesteps).normalize()

    if len(df_daymet) != len(timesteps):
        raise ValueError(
            "Daymet tensor and timesteps lengths do not match"
        )

    df_daymet = df_daymet[
        (df_daymet["date"] >= hist_start) &
        (df_daymet["date"] < pred_date)
    ].copy()

    if df_daymet.empty:
        raise ValueError(
            f"No Daymet data available between "
            f"{hist_start.date()} and {pred_date.date()}"
        )

    # --------------------------------------------------
    # 2. Find GEFS initialization files in window
    # --------------------------------------------------
    nc_files = []

    for day in pd.date_range(
        hist_start,
        pred_date - pd.Timedelta(days=1),
        freq="D",
    ):

        init_str = day.strftime("%Y%m%d00")

        f_path = os.path.join(
            GEFS_DIR,
            f"gefs_processed_{ens_name}",
            f"gefs_processed.{init_str}.{ens_name}.nc",
        )

        if os.path.exists(f_path):
            nc_files.append(f_path)

    if not nc_files:
        raise FileNotFoundError(
            f"No GEFS files found in previous {window} days"
        )

    # --------------------------------------------------
    # 3. Collect historical GEFS forecasts
    # --------------------------------------------------
    records = []

    for nc_file in nc_files:

        with xr.open_dataset(nc_file) as ds:

            basin = ds.sel({
                "catchment-id": basin_id
            })

            dates_g = pd.DatetimeIndex(
                pd.to_datetime(basin["time"].values)
            ).normalize()

            values = {
                var: basin[var].values
                for var in variables
            }

            for i, valid_date in enumerate(dates_g):

                # Use only valid dates in historical window
                if not (hist_start <= valid_date < pred_date):
                    continue

                records.append({
                    "date": valid_date,
                    "lead": i + 1,
                    "prcp": float(values["prcp"][i]),
                    "tmean": float(values["tmean"][i]),
                    "pet": float(values["pet"][i]),
                })

    if not records:
        raise ValueError(
            "No GEFS forecast values available "
            "within the historical window"
        )

    df_gefs = pd.DataFrame(records)

    # --------------------------------------------------
    # 4. Calculate lead-specific corrections
    # --------------------------------------------------
    corrections = {}

    for lead in range(1, 11):

        gefs_lead = df_gefs[
            df_gefs["lead"] == lead
        ]

        corrections[lead] = {}

        for var in variables:

            vals_d = df_daymet[var].dropna().to_numpy()
            vals_g = gefs_lead[var].dropna().to_numpy()

            # Require at least 10 values from each dataset
            if len(vals_d) < 10 or len(vals_g) < 10:
                continue

            if method == "scalar":

                mean_d = vals_d.mean()
                mean_g = vals_g.mean()

                if var == "tmean":
                    correction = mean_d - mean_g
                else:
                    correction = (
                        mean_d / mean_g
                        if mean_g > 1e-6 else 1.0
                    )

            else:  # CDF

                percentiles = np.linspace(0, 100, 101)

                q_gefs = np.percentile(
                    vals_g, percentiles
                )

                q_daymet = np.percentile(
                    vals_d, percentiles
                )

                # Remove duplicate GEFS quantiles
                unique_x, inverse = np.unique(
                    q_gefs,
                    return_inverse=True,
                )

                unique_y = np.full(
                    len(unique_x),
                    -np.inf,
                )

                np.maximum.at(
                    unique_y,
                    inverse,
                    q_daymet,
                )

                correction = (unique_x, unique_y)

            corrections[lead][var] = correction

    return corrections


def pre_processing(df, corrections, method="scalar"):
    """
    Apply lead-specific GEFS bias corrections.

    Supports:
        scalar: mean ratio / additive temperature correction
        cdf: empirical quantile mapping
    """

    if method not in ("scalar", "cdf"):
        raise ValueError(f"Unknown correction method: {method}")

    df = df.copy()

    # Forecast leads: 1, 2, ..., horizon
    if "lead" not in df.columns:
        df["lead"] = np.arange(1, len(df) + 1)

    for idx, row in df.iterrows():

        lead = int(row["lead"])

        for var in ["prcp", "tmean", "pet"]:

            correction = corrections.get(
                lead, {}
            ).get(var)

            # No correction available: keep original value
            if correction is None:
                continue

            value = row[var]

            if pd.isna(value):
                continue

            if method == "scalar":

                if var == "tmean":
                    corrected = value + correction
                else:
                    corrected = value * correction

            elif method == "cdf":

                q_in, q_out = correction

                corrected = np.interp(
                    value,
                    q_in,
                    q_out,
                )

                # Preserve zero-precipitation forecasts
                if var == "prcp" and value <= 0:
                    corrected = 0.0

            # Physical constraints
            if var in ("prcp", "pet"):
                corrected = max(0.0, corrected)

            df.at[idx, var] = corrected

    return df

   
def post_processing(sim_pre_GEFS, ens_preds): # post processing
    """
    Apply a constant offset correction so that the first forecasted streamflow
    connects smoothly with the last simulated (Daymet) streamflow.

    Parameters
    ----------
    sim_pre_GEFS : np.ndarray
        1D array of simulated streamflow before the forecast period (from HBV using Daymet).
    ens_preds : np.ndarray
        2D array of ensemble forecasts with shape (N_ENSEMBLES, FORECAST_LENGTH).

    Returns
    -------
    np.ndarray
        Offset-corrected ensemble forecasts of the same shape.
    """
    Q_sim_end = sim_pre_GEFS[-1]  # last simulated streamflow value
    ens_preds_corrected = []

    for ens in ens_preds:
        offset = Q_sim_end - ens[0]          # compute offset
        ens_corrected = np.maximum(ens + offset, 0.0)  # apply and ensure non-negative
        ens_preds_corrected.append(ens_corrected)

    return np.array(ens_preds_corrected)


def obtain_gage_name(GAGE_NAME_PATH, gage_id): 
    gage_names = pd.read_csv(
    GAGE_NAME_PATH,
    sep=";",
    dtype={"gauge_id": str}
    ) 
    gage_names['gauge_id'] = gage_names['gauge_id'].str.lstrip('0') 
    match = gage_names.loc[gage_names['gauge_id'] == str(gage_id), 'gauge_name']
    if match.empty:
        raise ValueError(f"GAGE_ID {gage_id} not found in {GAGE_NAME_PATH}")
    else:
        gage_name = match.values[0]
    return gage_name

  
def GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date, showblock=False):
    if not idx_list: raise ValueError(f"Starting GEFS date {start_date.date()} not exist!")
    if len(fc_block) < horizon: raise ValueError("Extracted GEFS forcing not enough!")
    if showblock:
        print("\n========= GEFS Forecast Block (NO ERROR) =========")
        print(fc_block.to_string(index=False))
         
def run_warm_forecasts_restart(
    model,
    state_path,
    gage_id,
    basin_idx,
    start_date,
    horizon,
    N_ENSEMBLES,
    GEFS_DIR,
    data_loader,
    timesteps,
    WINDOW,
    CORRECTION,
    device,
    verbose,
):
    ens_preds = []

    name = list(model.model_dict.keys())[0]
    nn_model = model.model_dict[name].nn_model
    phy_model = model.model_dict[name].phy_model

    # restart behavior
    phy_model.cache_states = True
    phy_model.warm_up = 0
    phy_model.warm_up_states = True

    for ens_id in range(N_ENSEMBLES): 
        # 1. Reset to saved stop-point states 
        model.load_states(path=state_path)

        # 2. Read GEFS forcings
        ens_num = ens_id + 1
        init_date = pd.Timestamp(start_date).normalize()
        init_str = init_date.strftime("%Y%m%d00")

        f_path = os.path.join(
            GEFS_DIR,
            f"gefs_processed_ens{ens_num:02d}",
            f"gefs_processed.{init_str}.ens{ens_num:02d}.nc"
        )

        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Missing GEFS nc file: {f_path}")
         
        with xr.open_dataset(f_path) as ds:

            # Select basin using 8-digit string ID
            basin_id = str(gage_id).zfill(8)
            basin_ds = ds.sel({"catchment-id": basin_id})

            # Check available forecast days
            available_days = basin_ds.sizes["time"]

            if horizon > available_days:
                raise ValueError(
                    f"Requested {horizon} forecast days, "
                    f"but only {available_days} are available."
                )

            # Select forecast horizon
            basin_ds = basin_ds.isel(time=slice(0, horizon))

            # Convert to DataFrame
            fc_block = (
                basin_ds[["prcp", "tmean", "pet"]]
                .to_dataframe()
                .reset_index()
            )
        
        # Convert timestamps to daily dates
        fc_block["date"] = pd.to_datetime(
            fc_block["time"]
        ).dt.normalize()

        # Add forecast lead (1 to horizon)
        fc_block["lead"] = np.arange(1, len(fc_block) + 1)

        # Check forecast length
        if len(fc_block) != horizon:
            raise ValueError(
                f"Expected {horizon} days, got {len(fc_block)}"
            )

        # Verify first forecast date
        if fc_block["date"].iloc[0] != init_date:
            raise ValueError(
                f"Forecast starts on {fc_block['date'].iloc[0]}, "
                f"but initialization date is {init_date}"
            )
        
        print("end here")

        # 3. Bias correction
        if CORRECTION:
            bias_corrections = compute_bias_correction_from_dataset(
                data_loader.dataset["x_phy"],
                GEFS_DIR,
                timesteps,
                basin_idx,
                gage_id,
                ens_id,
                start_date=start_date,
                window=WINDOW,
                method=CORRECTION,
            )
        fc_block = pre_processing(fc_block, bias_corrections, method=CORRECTION)
        
        # 4. Build HBV forcing tensor 
        raw_np = fc_block[["prcp", "tmean", "pet"]].to_numpy().astype(np.float32)
        x_phy_fc = torch.tensor(raw_np[:, np.newaxis, :], dtype=torch.float32, device=device)
        x_phy_fc[torch.isnan(x_phy_fc)] = 0.0

        # ----------------------------------------
        # 5. Build NN input tensor
        # ----------------------------------------
        xc_nn_norm_fc = build_xc_nn_norm_forecast(
            data_loader=data_loader,
            basin_idx=basin_idx,
            fc_block=fc_block,
            device=device,
            verbose = verbose,
        )

        # ----------------------------------------
        # 6. Run LSTM -> HBV
        # ----------------------------------------
        with torch.no_grad():
            raw_nn_out = nn_model(xc_nn_norm_fc)
            fluxes = phy_model(
                x_dict={"x_phy": x_phy_fc},
                parameters=raw_nn_out
            )

        q_fc = fluxes["streamflow"][:, 0, 0].detach().cpu().numpy()
        ens_preds.append(q_fc)

    return np.array(ens_preds)

def build_xc_nn_norm_forecast(data_loader, basin_idx, fc_block, device, verbose):
    if verbose:
        print("===== BUILD XC_NN_NORM FORECAST =====")

    # 1. Raw forecast forcings
    x_nn_fc = fc_block[data_loader.nn_forcings].to_numpy().astype(np.float32)   # [T, 3]
    if verbose:
        print("raw GEFS forcings shape:", x_nn_fc.shape)

    x_nn_fc = x_nn_fc[:, np.newaxis, :]                                          # [T, 1, 3]
    if verbose:
        print("after add basin dim:", x_nn_fc.shape)

    # 2. Static attributes
    c_nn_fc = data_loader.dataset["c_nn"][basin_idx:basin_idx+1].detach().cpu().numpy().astype(np.float32)  # [1, 35]
    if verbose:
        print("static attributes shape:", c_nn_fc.shape)

    # 3. Normalize forcings
    x_nn_norm = data_loader.to_norm(x_nn_fc, data_loader.nn_forcings)
    if verbose:
        print("normalized forcings shape:", x_nn_norm.shape)

    # 4. Normalize static attrs
    c_nn_norm = data_loader.to_norm(c_nn_fc, data_loader.nn_attributes)
    if verbose:
        print("normalized static attrs shape:", c_nn_norm.shape)

    # 5. Repeat static attrs over time
    c_nn_norm = np.repeat(
        np.expand_dims(c_nn_norm, 0),
        x_nn_norm.shape[0],
        axis=0
    )  # [T, 1, 35]
    if verbose:
        print("repeated static attrs shape:", c_nn_norm.shape)

    # 6. Concatenate
    xc_nn_norm_fc = np.concatenate((x_nn_norm, c_nn_norm), axis=2)  # [T, 1, 38]
    if verbose:
        print("final xc_nn_norm_fc shape:", xc_nn_norm_fc.shape)

    # 7. Convert to tensor
    xc_nn_norm_fc = torch.tensor(xc_nn_norm_fc, dtype=torch.float32, device=device)
    if verbose:
        print("final tensor shape:", xc_nn_norm_fc.shape)

    if verbose:
        print("=====================================")

    return xc_nn_norm_fc

 
def run_segment(
    model,
    dataset,
    basin_idx,
    device,
    name=None,
    start_idx=None,
    end_idx=None,
    warm_up=0,
    cache_states=False,
    warm_up_states=True,
    state_path=None,
):
    """
    Slice one basin/time segment from dataset and run NN + physical model.

    Parameters
    ----------
    model : ModelHandler
    dataset : dict
        Must contain 'xc_nn_norm' and 'x_phy'
    basin_idx : int
    device : torch.device
    name : str or None
        Model name. If None, uses the first key in model.model_dict.
    start_idx, end_idx : int or None
        Time slice for the segment.
    warm_up : int
    cache_states : bool
    warm_up_states : bool
    state_path : str or None
        If provided, load saved states before forward.

    Returns
    ------- 
    fluxes : dict 
    """
    if name is None:
        name = list(model.model_dict.keys())[0]

    nn_model = model.model_dict[name].nn_model
    phy_model = model.model_dict[name].phy_model

    data_dict = {
        "xc_nn_norm": dataset["xc_nn_norm"][start_idx:end_idx, basin_idx:basin_idx + 1, :].clone(),
        "x_phy": dataset["x_phy"][start_idx:end_idx, basin_idx:basin_idx + 1, :].clone(),
    }

    phy_model.warm_up = warm_up
    phy_model.cache_states = cache_states
    phy_model.warm_up_states = warm_up_states

    model.eval()

    if state_path is not None:
        model.load_states(path=state_path)

    with torch.no_grad():
        xc = data_dict["xc_nn_norm"].to(device)
        x_phy = data_dict["x_phy"].to(device)

        nn_out = nn_model(xc)
        fluxes = phy_model(
            x_dict={"x_phy": x_phy},
            parameters=nn_out,
        )
    # return nn_out, fluxes, phy_model, data_dict
    return fluxes['streamflow'][:, 0, 0].detach().cpu().numpy()


def selectbasins():
    with open("./predownloaded/Best_two_in_region.json", "r") as f:
        basin_groups = json.load(f)
    basin_pool = [b for group in basin_groups.values() for b in group] 
    return basin_pool


def startid_endid(config, warm, ENSEMBLE_START_DATE, FORECAST):
    timesteps = Dates(config['sim'], config['model']['rho']).batch_daily_time_range # 730
    # print("check on config", config["sim"])
    sidx = np.where(timesteps == ENSEMBLE_START_DATE)[0][0] # 715
    eidx = sidx + FORECAST # 730 
    history_len = len(timesteps) - warm - FORECAST
    # print("sidx, eidx, timesteps, history_len are", sidx, eidx, timesteps, history_len)
    return sidx, eidx, timesteps, history_len 
 

def evaluate_one_basin(model, dataset, basin_idx, device, nmul, staind, tdRep, routing, dydrop):
    #  
    x_phy = torch.nan_to_num(
        dataset["x_phy"][:, basin_idx:basin_idx+1, :3],
        nan=0.0
    ).to(device)     # [T, 1, 3]

    # ---------------- Observed discharge ----------------
    obs = dataset["target"][:, basin_idx, 0].cpu().numpy()   # [T]

    # ---------------- NN → HBV parameters ----------------
    hist_dict = {
        "xc_nn_norm": dataset["xc_nn_norm"][:, basin_idx:basin_idx+1].clone()
        # [T, 1, 38]
    }

    pars, rtwts = get_parameters_from_model(
        model.model_dict["Hbv_1_1p"],
        hist_dict,
        n_par=14,
        mu=nmul,
        device=device,
    )
    
    hbv = h1pp.HBV().to(device) 
    # ---------------- Run YOUR HBV1_1p ----------------
    with torch.no_grad():
        Qs = hbv(
            x=x_phy,          # [T, 1, 3]
            parameters=pars,  # [T, 1, 14, 16]
            staind=staind,
            tdlst=tdRep,
            mu=nmul,
            muwts=None,
            rtwts=rtwts,      # [1,2]
            bufftime=0,
            instate=False,
            outstate=False,
            routOpt=routing,
            dydrop=dydrop,
        )

    sim = Qs[:, 0, 0].detach().cpu().numpy()

    nse_val = nse(sim, obs)
    kge_val = kge(sim, obs)
    return nse_val, kge_val

def print_selected_basin_metrics_from_json(metrics, selected_basins, basin_pool):
    sumnse, sumkge = 0.0, 0.0
    count = 0

    for GAGE_ID in selected_basins:
        print(f"For basin {GAGE_ID} ===")
        basin_idx = basin_pool.index(GAGE_ID)

        nse_val = metrics['nse'][basin_idx]
        kge_val = metrics['kge'][basin_idx]
        print(f"NSE={nse_val}, KGE={kge_val}")

        if nse_val is None or kge_val is None:
            print(f"Skip basin {GAGE_ID} because metric is None")
            continue

        sumnse += float(nse_val)
        sumkge += float(kge_val)
        count += 1

    if count > 0:
        print(f"\nAverage over {count} valid selected basins: NSE={sumnse/count}, KGE={sumkge/count}")
    else:
        print("\nNo valid basins found.")
        

def cleannans(metrics, METRIC):

    metric_vals = metrics[METRIC]

    clean_metric_vals = [
        float(x) for x in metric_vals
        if x is not None and not np.isnan(float(x))
    ]

    metrics_clean = {METRIC: clean_metric_vals}
    return metrics_clean

    
def plot_forecast_separate(
    GAGE_NAME_PATH,
    gage_id,
    start_date,
    obs,                 # length = history_len + HORIZON
    sim,                 # length = history_len
    ens_preds,           # shape = [n_ens, HORIZON]
    history_len,
    Q_all_np,            # continuous output from long run
    sidx,                # input index of forecast start in the dataset timeline
    FORECAST,
    WARMUPTIME,
    det_pred=None,       # length = HORIZON
    save_dir=None,
    file_prefix=None,
    plot_history_days=60,
):
    """
    Save three separate figures:
      1) Top forecast figure (full history + forecast)
      2) Bottom continuous-run figure
      3) Cleaner zoomed forecast figure (plot_history_days + FORECAST)

    Notes
    -----
    HBV restart is plotted with a connector segment from the last history
    simulation point to the first restart point, without forcing the values
    to be equal.
    """
    gage_name = obtain_gage_name(GAGE_NAME_PATH, gage_id)

    ens_preds = np.asarray(ens_preds)
    if ens_preds.ndim != 2:
        raise ValueError(f"ens_preds must be [n_ens, HORIZON], got {ens_preds.shape}")

    HORIZON = ens_preds.shape[1]
    if HORIZON != FORECAST:
        print(f"WARNING: HORIZON={HORIZON} != FORECAST={FORECAST}")

    start_date = pd.to_datetime(start_date)

    # ---------- Top plot dates ----------
    dates_obs_top = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len + HORIZON,
    )
    dates_sim_top = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len,
    )

    # ---------- Pad ensembles ----------
    padded_ensembles = np.full((ens_preds.shape[0], history_len + HORIZON), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    ens_min = np.full(history_len + HORIZON, np.nan)
    ens_max = np.full(history_len + HORIZON, np.nan)
    if not np.all(np.isnan(ens_preds)):
        ens_min[history_len:] = np.nanmin(ens_preds, axis=0)
        ens_max[history_len:] = np.nanmax(ens_preds, axis=0)

    # ---------- Continuous run dates ----------
    dataset_day0 = start_date - pd.Timedelta(days=sidx)
    dates_cont = pd.date_range(
        start=dataset_day0 + pd.Timedelta(days=WARMUPTIME),
        periods=len(Q_all_np),
    )

    # ---------- x-ranges ----------
    pad_days = 3

    # full range for original plot 1 and continuous plot 2
    x0, x1 = dates_obs_top[0], dates_obs_top[-1]

    # zoomed range for plot 3
    plot_history_days = min(plot_history_days, history_len)
    x0_zoom = start_date - pd.Timedelta(days=plot_history_days)
    x1_zoom = start_date + pd.Timedelta(days=FORECAST - 1)

    # ---------- Convert arrays ----------
    obs = np.asarray(obs).reshape(-1)
    sim = np.asarray(sim).reshape(-1)
    ens_preds = np.asarray(ens_preds)

    # Split observed into history and forecast parts
    obs_hist = obs[:history_len]
    obs_fcst = obs[history_len:history_len + HORIZON]

    # --- Pre-forecast metrics: sim vs observed history ---
    pre_nse, pre_kge = np.nan, np.nan
    valid_pre = np.isfinite(sim) & np.isfinite(obs_hist)

    if valid_pre.sum() > 1:
        pre_nse = nse(sim[valid_pre], obs_hist[valid_pre])
        pre_kge = kge(sim[valid_pre], obs_hist[valid_pre])

    # --- Forecast ensemble metrics: each ensemble member vs observed forecast ---
    ens_nse_list = []
    ens_kge_list = []

    for i in range(ens_preds.shape[0]):
        pred_i = ens_preds[i]
        valid_i = np.isfinite(pred_i) & np.isfinite(obs_fcst)

        if valid_i.sum() > 1:
            ens_nse_i = nse(pred_i[valid_i], obs_fcst[valid_i])
            ens_kge_i = kge(pred_i[valid_i], obs_fcst[valid_i])
        else:
            ens_nse_i = np.nan
            ens_kge_i = np.nan

        ens_nse_list.append(ens_nse_i)
        ens_kge_list.append(ens_kge_i)

    ens_nse_arr = np.asarray(ens_nse_list)
    ens_kge_arr = np.asarray(ens_kge_list)

    metric_text = (
        f"Pre-Forecast\n"
        f"NSE={pre_nse:.3f}, KGE={pre_kge:.3f}\n\n"
        f"GEFS Ensembles\n"
    )
    for i in range(len(ens_nse_arr)):
        metric_text += f"E{i+1}: NSE={ens_nse_arr[i]:.3f}, KGE={ens_kge_arr[i]:.3f}\n"

    # ---------- Restart dates for connector-style plotting ----------
    has_det = det_pred is not None
    if has_det:
        det_pred = np.asarray(det_pred).reshape(-1)
        if det_pred.shape[0] != HORIZON:
            raise ValueError(f"det_pred length {det_pred.shape[0]} != HORIZON {HORIZON}")

        restart_dates = pd.date_range(start=start_date, periods=HORIZON)
        connector_dates = [dates_sim_top[-1], restart_dates[0]]
        connector_vals = [sim[-1], det_pred[0]]

    # ==========================================================
    # FIGURE 1: TOP FORECAST (FULL RANGE)
    # ==========================================================
    fig1, ax1 = plt.subplots(figsize=(12, 5.5))

    ax1.plot(dates_obs_top, obs, "k-", lw=2, label="Observed")
    ax1.plot(dates_sim_top, sim, "r-", lw=1.5, label="Simulation (history)")

    ax1.fill_between(
        dates_obs_top, ens_min, ens_max,
        color="green", alpha=0.1, label="Ensembles"
    )
    for i in range(ens_preds.shape[0]):
        ax1.plot(dates_obs_top, padded_ensembles[i], lw=1.5, color="darkgreen")

    if has_det:
        # connector: last history point -> first restart point
        ax1.plot(
            connector_dates,
            connector_vals,
            color="purple",
            lw=1.5
        )
        # actual restart line
        ax1.plot(
            restart_dates,
            det_pred,
            color="purple",
            lw=1.5,
            label="HBV Restart"
        )

    ax1.axvspan(
        dates_obs_top[0], dates_obs_top[history_len],
        color="gray", alpha=0.2, label="Pre-Forecast"
    )
    ax1.axvspan(
        dates_obs_top[history_len], dates_obs_top[-1],
        color="orange", alpha=0.2, label="Forecast"
    )

    ax1.text(
        0.98, 0.98,
        metric_text,
        transform=ax1.transAxes,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax1.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))
    ax1.set_title(f"GEFS Forecast — Gage {gage_id} ({gage_name})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Streamflow (mm/day)")
    ax1.grid(True, linestyle="--", linewidth=1.5, alpha=0.7)

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="upper center", ncol=max(1, int(np.ceil(len(l1) / 4))))
    fig1.tight_layout()

    # ==========================================================
    # FIGURE 2: BOTTOM CONTINUOUS RUN
    # ==========================================================
    fig2, ax2 = plt.subplots(figsize=(12, 4.8))

    mask = (dates_cont >= x0) & (dates_cont <= x1)
    cont_dates_win = dates_cont[mask]
    cont_vals_win = np.asarray(Q_all_np)[mask]

    # ---------- Align observed data with continuous run ----------
    obs_series = pd.Series(obs, index=dates_obs_top)
    cont_series = pd.Series(cont_vals_win, index=cont_dates_win)

    common_dates = obs_series.index.intersection(cont_series.index)
    obs_aligned = obs_series.loc[common_dates].values
    cont_aligned = cont_series.loc[common_dates].values

    valid = np.isfinite(obs_aligned) & np.isfinite(cont_aligned)

    nse_val, kge_val = np.nan, np.nan
    if valid.sum() > 1:
        obs_valid = obs_aligned[valid]
        cont_valid = cont_aligned[valid]

        ax2.plot(
            common_dates[valid],
            obs_valid,
            "k-",
            lw=1.5,
            markersize=6,
            label="Observed"
        )

        nse_val = nse(cont_valid, obs_valid)
        kge_val = kge(cont_valid, obs_valid)

    ax2.plot(
        cont_dates_win,
        cont_vals_win,
        "-",
        color="red",
        lw=1.5,
        markersize=6,
        label="Continuous Simulation"
    )

    if has_det:
        ax2.plot(
            connector_dates,
            connector_vals,
            color="purple",
            lw=1.5
        )
        ax2.plot(
            restart_dates,
            det_pred,
            color="purple",
            lw=1.5,
            label="HBV Restart"
        )

    ax2.text(
        0.98, 0.98,
        f"NSE = {nse_val:.3f}\nKGE = {kge_val:.3f}",
        transform=ax2.transAxes,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax2.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))
    ax2.set_title(f"Continuous Run — Gage {gage_id} ({gage_name})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Streamflow (mm/day)")
    ax2.grid(True, linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.legend(loc="upper center")
    fig2.tight_layout()

    # ==========================================================
    # FIGURE 3: ZOOMED FORECAST (plot_history_days + FORECAST)
    # ==========================================================
    fig3, ax3 = plt.subplots(figsize=(12, 5.5))

    ax3.plot(dates_obs_top, obs, "k-", lw=2, label="Observed")
    ax3.plot(dates_sim_top, sim, "r-", lw=1.5, label="Simulation (history)")

    ax3.fill_between(
        dates_obs_top, ens_min, ens_max,
        color="green", alpha=0.1, label="Ensembles"
    )
    for i in range(ens_preds.shape[0]):
        ax3.plot(dates_obs_top, padded_ensembles[i], lw=1.5, color="darkgreen")

    if has_det:
        ax3.plot(
            connector_dates,
            connector_vals,
            color="purple",
            lw=1.5
        )
        ax3.plot(
            restart_dates,
            det_pred,
            color="purple",
            lw=1.5,
            label="HBV Restart"
        )

    ax3.axvspan(x0_zoom, start_date, color="gray", alpha=0.2, label="Pre-Forecast")
    ax3.axvspan(start_date, x1_zoom, color="orange", alpha=0.2, label="Forecast")

    ax3.text(
        0.98, 0.98,
        metric_text,
        transform=ax3.transAxes,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax3.set_xlim(
        x0_zoom - pd.Timedelta(days=pad_days),
        x1_zoom + pd.Timedelta(days=pad_days)
    ) 
    # ---------- Auto y-range for zoomed window ----------
    y_candidates = []

    mask_obs_zoom = (dates_obs_top >= x0_zoom) & (dates_obs_top <= x1_zoom)
    if np.any(mask_obs_zoom):
        y_candidates.append(obs[mask_obs_zoom])

    mask_sim_zoom = (dates_sim_top >= x0_zoom) & (dates_sim_top <= x1_zoom)
    if np.any(mask_sim_zoom):
        y_candidates.append(sim[mask_sim_zoom])

    for i in range(ens_preds.shape[0]):
        ens_series_i = padded_ensembles[i]
        if np.any(mask_obs_zoom):
            y_candidates.append(ens_series_i[mask_obs_zoom])

    if has_det:
        mask_det_zoom = (restart_dates >= x0_zoom) & (restart_dates <= x1_zoom)
        if np.any(mask_det_zoom):
            y_candidates.append(det_pred[mask_det_zoom])
        y_candidates.append(np.asarray(connector_vals))

    if len(y_candidates) > 0:
        y_all = np.concatenate([np.asarray(y).ravel() for y in y_candidates])
        y_all = y_all[np.isfinite(y_all)]

        if y_all.size > 0:
            y_min = np.min(y_all)
            y_max = np.max(y_all)

            if y_max > y_min:
                y_pad = 0.08 * (y_max - y_min)
            else:
                y_pad = max(0.1, 0.08 * max(abs(y_max), 1.0))

            ax3.set_ylim(max(0, y_min - y_pad), y_max + y_pad)


    ax3.set_title(
        f"GEFS Forecast (Zoomed {plot_history_days}-Day History) — "
        f"Gage {gage_id} ({gage_name})"
    )
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Streamflow (mm/day)")
    ax3.grid(True, linestyle="--", linewidth=1.5, alpha=0.7)

    h3, l3 = ax3.get_legend_handles_labels()
    ax3.legend(h3, l3, loc="upper center", ncol=max(1, int(np.ceil(len(l3) / 4))))
    fig3.tight_layout()

    # ---------- Save ----------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if file_prefix is None:
            file_prefix = f"GAGE_{gage_id}"

        top_path = os.path.join(save_dir, f"{file_prefix}_forecast.png")
        bottom_path = os.path.join(save_dir, f"{file_prefix}_continuous.png")
        zoom_path = os.path.join(save_dir, f"{file_prefix}_forecast_zoom.png")

        fig1.savefig(top_path, dpi=300, bbox_inches="tight")
        fig2.savefig(bottom_path, dpi=300, bbox_inches="tight")
        fig3.savefig(zoom_path, dpi=300, bbox_inches="tight")

        print(f"Saved top figure to: {top_path}")
        print(f"Saved bottom figure to: {bottom_path}")
        print(f"Saved zoomed figure to: {zoom_path}")

    plt.show()
    print(f"Saved plot for basin {gage_id} → {save_dir}")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)