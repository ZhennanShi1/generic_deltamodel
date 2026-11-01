# ============================================================
# Eq. (5): CRPS for one forecast case
# ============================================================

def crps_single(esp, obs):
    """
    esp : shape (n_members,)
        Ensemble forecast for one basin, one init date, one lead time (LT).
    obs : scalar
        Observed/reference streamflow at target date.
    """
    esp = np.asarray(esp, dtype=float)

    if not np.isfinite(obs) or not np.all(np.isfinite(esp)):
        return np.nan

    term1 = np.mean(np.abs(esp - obs))
    term2 = 0.5 * np.mean(
        np.abs(esp[:, None] - esp[None, :])
    )

    return term1 - term2
# ============================================================
# CRPS_forecast(LT): Average CRPS_forecast for one basin and one lead time
# ============================================================

def average_crps(esps, obs):
    """ 
    esps : shape (n_members, n_init_dates) n_init_dates is 362 initialization dates
        Ensemble forecasts for one basin and one lead time. LT is fixed as one LT.
    obs : shape (n_init_dates,)
        Reference streamflow at target dates.
    """
    esps = np.asarray(esps, dtype=float) # (5, 362)
    obs = np.asarray(obs, dtype=float) # (362,)

    crps_values = []

    for t in range(len(obs)): # go through all initialization date
        esp_t = esps[:, t] # this is for all ensembles, one initialzation date, one LT
        obs_t = obs[t]

        crps_values.append(
            crps_single(esp_t, obs_t)
        )

    return np.nanmean(crps_values) # take avg of all initialization dates, one LT

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
    init_indices = np.asarray(init_indices, dtype=int)

    # GEFS Day 1 = initialization date
    target_indices = init_indices + lead_time - 1

    # Last available observation before initialization
    persistence_indices = init_indices - 1

    if np.any(persistence_indices < 0):
        raise ValueError("Missing pre-initialization discharge")

    if np.any(target_indices >= len(qsim)):
        raise ValueError("Target dates exceed discharge record")

    target_q = qsim[target_indices]
    initial_q = qsim[persistence_indices]

    return np.nanmean(np.abs(target_q - initial_q))



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
    max_lead=10
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
            target_indices = init_indices + lt - 1
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
    
    

# ============================================================
# GEFS CRPSS: SINGLE OR MULTIPLE BASINS
# ============================================================ 
import warnings

def _gefs_ensemble_crps(forecast, reference):
    """
    forecast: (n_dates, n_leads, n_members)
    reference: (n_dates, n_leads)
    """
    term1 = np.mean(
        np.abs(forecast - reference[:, :, None]), axis=2
    )
    term2 = 0.5 * np.mean(
        np.abs(
            forecast[:, :, :, None] -
            forecast[:, :, None, :]
        ),
        axis=(2, 3)
    )
    return term1 - term2


def _gefs_basin_crpss(
    basin_id,
    basin_results,
    save_root,
    year,
    n_leads,
    n_ensembles,
    lead1_offset,
    min_valid_cases,
    reference="observed",
    climatology_years=None,
):
    """
    Calculate lead-wise CRPSS for one basin.

    reference:
        "observed"  -> basin_results["obs"]
        "simulated" -> basin_results["sim"]

    climatology_years:
        Optional years used to calculate climatological mean.
        Example: range(2009, 2012).
        None uses the complete available reference series.

    Forecast NPZ:
        forecast shape = (n_ensembles, n_leads)
    """

    basin_id = f"{int(basin_id):08d}"
    basin_dir = os.path.join(save_root, basin_id)

    files = sorted(glob.glob(
        os.path.join(basin_dir, "forecast_*.npz")
    ))

    forecasts_list = []
    init_dates_list = []

    # ====================================================
    # 1. Load saved GEFS forecasts
    # ====================================================

    for path in files:

        with np.load(path) as data:
            fc = np.asarray(
                data["forecast"], dtype=float
            )

            init_date = pd.Timestamp(
                str(data["init_date"].item())
            ).normalize()

        if init_date.year != year:
            continue

        if fc.shape != (n_ensembles, n_leads):
            raise ValueError(
                f"Invalid forecast shape {fc.shape}: {path}"
            )

        forecasts_list.append(fc.T)
        init_dates_list.append(init_date)

    if not forecasts_list:
        raise FileNotFoundError(
            f"No {year} forecasts for {basin_id}"
        )

    forecasts = np.stack(forecasts_list)
    init_dates = pd.DatetimeIndex(init_dates_list)

    if not init_dates.is_unique:
        raise ValueError(
            f"Duplicate forecast dates: {basin_id}"
        )

    # ====================================================
    # 2. Load observed or simulated streamflow
    # ====================================================

    basin_data = next(
        (
            value
            for key, value in basin_results.items()
            if int(key) == int(basin_id)
        ),
        None
    )

    if basin_data is None:
        raise KeyError(
            f"Missing basin_results for {basin_id}"
        )

    if reference == "observed":
        data_key = "obs"
    elif reference == "simulated":
        data_key = "sim"
    else:
        raise ValueError(
            "reference must be 'observed' or 'simulated'"
        )

    qref = np.asarray(
        basin_data[data_key], dtype=float
    ).reshape(-1)

    qref_dates = pd.DatetimeIndex(
        basin_data["dates"]
    ).normalize()

    if len(qref) != len(qref_dates):
        raise ValueError(
            f"Reference length mismatch: {basin_id}"
        )

    if not qref_dates.is_unique:
        raise ValueError(
            f"Duplicate reference dates: {basin_id}"
        )

    qref_series = pd.Series(
        qref, index=qref_dates
    )

    # ====================================================
    # 3. Align forecast valid dates with observations
    # ====================================================

    # lead1_offset=0:
    # lead 1 = initialization date
    # lead 2 = initialization date + 1 day
    # ...

    offsets = np.arange(
        lead1_offset,
        lead1_offset + n_leads
    )

    valid_dates = (
        init_dates.values[:, None]
        + offsets[None, :].astype("timedelta64[D]")
    )

    target_flow = qref_series.reindex(
        pd.DatetimeIndex(valid_dates.ravel())
    ).to_numpy().reshape(valid_dates.shape)

    # Persistence uses observed discharge at initialization
    initial_flow = qref_series.reindex(
        init_dates - pd.Timedelta(days=1)
    ).to_numpy()

    # ====================================================
    # 4. Calculate climatological mean
    # ====================================================

    if climatology_years is None:
        clim_values = qref
    else:
        clim_mask = qref_dates.year.isin(
            list(climatology_years)
        )
        clim_values = qref[clim_mask]

    finite_clim = clim_values[
        np.isfinite(clim_values)
    ]

    if len(finite_clim) == 0:
        raise ValueError(
            f"No valid climatology data for {basin_id}"
        )

    q_mean = np.mean(finite_clim)

    # ====================================================
    # 5. Identify valid cases
    # ====================================================

    valid = (
        np.isfinite(forecasts).all(axis=2)
        & np.isfinite(target_flow)
        & np.isfinite(initial_flow[:, None])
    )

    n_valid = valid.sum(axis=0)

    # ====================================================
    # 6. Forecast CRPS
    # ====================================================

    crps_fc_cases = _gefs_ensemble_crps(
        forecasts,
        target_flow
    )

    # ====================================================
    # 7. Persistence reference
    # ====================================================

    crps_pf_cases = np.abs(
        target_flow - initial_flow[:, None]
    )

    # ====================================================
    # 8. Average discharge reference
    # ====================================================

    crps_ad_cases = np.abs(
        target_flow - q_mean
    )

    # Same verification cases for all three scores
    crps_fc_cases[~valid] = np.nan
    crps_pf_cases[~valid] = np.nan
    crps_ad_cases[~valid] = np.nan

    # ====================================================
    # 9. Average CRPS across initialization dates
    # ====================================================

    crps_fc = np.full(n_leads, np.nan)
    crps_pf = np.full(n_leads, np.nan)
    crps_ad = np.full(n_leads, np.nan)

    for j in range(n_leads):
        if n_valid[j] == 0:
            continue

        crps_fc[j] = np.mean(
            crps_fc_cases[valid[:, j], j]
        )

        crps_pf[j] = np.mean(
            crps_pf_cases[valid[:, j], j]
        )

        crps_ad[j] = np.mean(
            np.abs(finite_clim - q_mean)
        )

    # ====================================================
    # 10. Calculate CRPSS
    # ====================================================

    crpss_pf = np.full(n_leads, np.nan)
    crpss_ad = np.full(n_leads, np.nan)

    valid_pf = (
        (n_valid >= min_valid_cases)
        & np.isfinite(crps_fc)
        & np.isfinite(crps_pf)
        & (crps_pf > 0)
    )

    valid_ad = (
        (n_valid >= min_valid_cases)
        & np.isfinite(crps_fc)
        & np.isfinite(crps_ad)
        & (crps_ad > 0)
    )

    crpss_pf[valid_pf] = (
        1 - crps_fc[valid_pf] / crps_pf[valid_pf]
    )

    crpss_ad[valid_ad] = (
        1 - crps_fc[valid_ad] / crps_ad[valid_ad]
    )

    return {
        "basin_id": basin_id,
        "reference": reference,
        "CRPSS_pf": crpss_pf,
        "CRPSS_ad": crpss_ad,
        "CRPS_forecast": crps_fc,
        "CRPS_ref_pf": crps_pf,
        "CRPS_ref_ad": crps_ad,
        "n_valid": n_valid,
        "n_init_dates": len(init_dates),
        "climatology_mean": q_mean,
    }

 
def run_gefs_crpss(
    mode,
    config,
    basin_results,
    basin_pool=None,
    GAGE_ID=None,
    START_DATE="2012-01-01",
    END_DATE="2012-12-31",
    PRE="scalar",
    WINDOW=30,
    POST=True,
    YEAR=2012,
    N_LEADS=10,
    N_ENSEMBLES=5,
    LEAD1_OFFSET=0,
    MIN_VALID_CASES=30,
    REFERENCE="observed",
    CLIMATOLOGY_YEARS=None,
    show_plot=True,
    ):
    """
    GEFS CRPSS evaluation.

    mode:
        "single" or "multiple"

    REFERENCE:
        "observed" or "simulated"

    CLIMATOLOGY_YEARS:
        None -> use entire available reference period
        range(2009, 2012) -> use 2009-2011

    Uses existing basin_results and basin_pool.
    """

    mode = mode.lower().strip()
    REFERENCE = REFERENCE.lower().strip()

    if REFERENCE not in ("observed", "simulated"):
        raise ValueError(
            "REFERENCE must be 'observed' or 'simulated'"
        )

    # ====================================================
    # 1. Select basins
    # ====================================================

    if mode == "single":

        if GAGE_ID is None:
            raise ValueError(
                "GAGE_ID is required for single mode"
            )

        selected_basins = [int(GAGE_ID)]

    elif mode == "multiple":

        if basin_pool is None:
            raise ValueError(
                "basin_pool is required for multiple mode"
            )

        selected_basins = [
            int(bid) for bid in basin_pool
        ]

    else:
        raise ValueError(
            "mode must be 'single' or 'multiple'"
        )

    selected_basins = list(
        dict.fromkeys(selected_basins)
    )

    if not selected_basins:
        raise ValueError("No basins selected")

    # ====================================================
    # 2. Output directories
    # ====================================================

    save_root = os.path.join(
        config["sim_dir"],
        "GEFS_1year_forecasts",
        f"{START_DATE}_{END_DATE}_pre-{PRE}_w{WINDOW}_post-{POST}"
    )

    output_dir = os.path.join(
        save_root,
        f"CRPSS_{REFERENCE}_{mode}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # ====================================================
    # 3. Evaluate basins
    # ====================================================

    basin_crpss = []
    failed_basins = []

    print("Mode:", mode)
    print("Reference:", REFERENCE)
    print("Requested basins:", len(selected_basins))

    for i, bid in enumerate(selected_basins):

        basin_id = f"{bid:08d}"

        try:

            result = _gefs_basin_crpss(
                basin_id=bid,
                basin_results=basin_results,
                save_root=save_root,
                year=YEAR,
                n_leads=N_LEADS,
                n_ensembles=N_ENSEMBLES,
                lead1_offset=LEAD1_OFFSET,
                min_valid_cases=MIN_VALID_CASES,
                reference=REFERENCE,
                climatology_years=CLIMATOLOGY_YEARS,
            )

            basin_crpss.append(result)

            print(
                f"[{i+1}/{len(selected_basins)}] "
                f"{basin_id}: "
                f"{result['n_init_dates']} forecasts"
            )

        except Exception as e:

            failed_basins.append({
                "basin_id": basin_id,
                "error": str(e),
            })

            print(f"FAILED {basin_id}: {e}")

    if not basin_crpss:
        raise RuntimeError(
            "No basins evaluated successfully"
        )

    # ====================================================
    # 4. Assemble arrays
    # ====================================================

    pf_all = np.stack([
        r["CRPSS_pf"] for r in basin_crpss
    ])

    ad_all = np.stack([
        r["CRPSS_ad"] for r in basin_crpss
    ])

    leads = np.arange(1, N_LEADS + 1)

    # ====================================================
    # 5. Save basin-level CSV
    # ====================================================

    rows = []

    for result in basin_crpss:

        for j, lead in enumerate(leads):

            rows.append({
                "basin_id": result["basin_id"],
                "reference": REFERENCE,
                "lead_time": lead,
                "CRPSS_pf": result["CRPSS_pf"][j],
                "CRPSS_ad": result["CRPSS_ad"][j],
                "CRPS_forecast": result["CRPS_forecast"][j],
                "CRPS_ref_pf": result["CRPS_ref_pf"][j],
                "CRPS_ref_ad": result["CRPS_ref_ad"][j],
                "n_valid": result["n_valid"][j],
            })

    results_df = pd.DataFrame(rows)

    csv_path = os.path.join(
        output_dir,
        f"CRPSS_{mode}_{REFERENCE}_{YEAR}.csv"
    )

    results_df.to_csv(
        csv_path, index=False
    )

    failed_csv = os.path.join(
        output_dir,
        "failed_basins.csv"
    )

    pd.DataFrame(
        failed_basins,
        columns=["basin_id", "error"]
    ).to_csv(failed_csv, index=False)

    # ====================================================
    # 6. Spatial statistics
    # ====================================================

    def column_stat(arr, statistic, percentile=None):
        values = []

        for j in range(arr.shape[1]):

            col = arr[:, j]
            col = col[np.isfinite(col)]

            if len(col) == 0:
                values.append(np.nan)
            elif statistic == "mean":
                values.append(np.mean(col))
            elif statistic == "percentile":
                values.append(
                    np.percentile(col, percentile)
                )

        return np.asarray(values)

    pf_mean = column_stat(pf_all, "mean")
    ad_mean = column_stat(ad_all, "mean")

    pf_p05 = column_stat(
        pf_all, "percentile", 5
    )
    pf_p25 = column_stat(
        pf_all, "percentile", 25
    )
    pf_p75 = column_stat(
        pf_all, "percentile", 75
    )
    pf_p95 = column_stat(
        pf_all, "percentile", 95
    )

    ad_p25 = column_stat(
        ad_all, "percentile", 25
    )
    ad_p75 = column_stat(
        ad_all, "percentile", 75
    )

    summary_df = pd.DataFrame({
        "lead_time": leads,
        "CRPSS_pf_mean": pf_mean,
        "CRPSS_ad_mean": ad_mean,
        "CRPSS_pf_p05": pf_p05,
        "CRPSS_pf_p25": pf_p25,
        "CRPSS_pf_p75": pf_p75,
        "CRPSS_pf_p95": pf_p95,
        "CRPSS_ad_p25": ad_p25,
        "CRPSS_ad_p75": ad_p75,
        "n_basins_pf": np.isfinite(pf_all).sum(axis=0),
        "n_basins_ad": np.isfinite(ad_all).sum(axis=0),
    })

    summary_csv = os.path.join(
        output_dir,
        f"CRPSS_summary_{mode}_{REFERENCE}_{YEAR}.csv"
    )

    summary_df.to_csv(
        summary_csv, index=False
    )

    # ====================================================
    # 7. Plot
    # ====================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    if mode == "multiple" and len(basin_crpss) > 1:

        ax.fill_between(
            leads,
            pf_p05,
            pf_p95,
            color="lightgray",
            alpha=0.8,
            label="5–95% (persistence)"
        )

        ax.fill_between(
            leads,
            pf_p25,
            pf_p75,
            color="gray",
            alpha=0.65,
            label="25–75% (persistence)"
        )

        ax.plot(
            leads,
            ad_p25,
            color="black",
            linestyle=":",
            linewidth=1
        )

        ax.plot(
            leads,
            ad_p75,
            color="black",
            linestyle=":",
            linewidth=1,
            label="25–75% (average discharge)"
        )

    ax.plot(
        leads,
        pf_mean,
        color="black",
        linewidth=2,
        linestyle="-",
        label=r"CRPSS$_{pf}$"
    )

    ax.plot(
        leads,
        ad_mean,
        color="black",
        linewidth=2,
        linestyle="--",
        label=r"CRPSS$_{ad}$"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    if mode == "single":

        basin_id = basin_crpss[0]["basin_id"]

        ax.set_title(
            f"GEFS Forecast Skill Against Observed\n"
            f"Across {len(basin_crpss)} Basins ({START_DATE} to {END_DATE})"
        )

        fig_name = (
            f"CRPSS_{basin_id}_{REFERENCE}_{YEAR}.png"
        )

    else:
 
        ax.set_title(
            f"GEFS Forecast Skill Against Observed\n"
            f"Across {len(basin_crpss)} Basins ({START_DATE} to {END_DATE})"
        )

        fig_name = (
            f"CRPSS_spatial_{REFERENCE}_{YEAR}.png"
        )

    ax.set_xlabel("Lead time [days]")
    ax.set_ylabel("CRPSS")

    ax.set_xticks(leads)

    # Do not restrict y-axis to [0, 1]:
    # Negative CRPSS values are meaningful.

    ax.legend(
        frameon=False,
        fontsize=9
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    fig_path = os.path.join(
        output_dir,
        fig_name
    )

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # ====================================================
    # 8. Return
    # ====================================================

    print("\nEvaluated:", len(basin_crpss))
    print("Failed:", len(failed_basins))
    print("Figure:", fig_path)
    print("CSV:", csv_path)

    return {
        "mode": mode,
        "reference": REFERENCE,
        "basin_crpss": basin_crpss,
        "crpss_pf_all": pf_all,
        "crpss_ad_all": ad_all,
        "results_df": results_df,
        "summary_df": summary_df,
        "failed_basins": failed_basins,
        "figure_path": fig_path,
        "csv_path": csv_path,
        "summary_csv": summary_csv,
        "output_dir": output_dir,
    }


def _load_gefs_metric_data(
    GAGE_ID,
    config,
    basin_results,
    START_DATE="2012-01-01",
    END_DATE="2012-12-31",
    PRE="scalar",
    WINDOW=30,
    POST=True,
    N_LEADS=10,
    N_ENSEMBLES=5,
    LEAD1_OFFSET=0,
    REFERENCE="observed",
):
    """
    Load saved GEFS NPZ files and align forecasts with reference flow.

    Returns:
        ensemble_mean: (n_dates, n_leads)
        target_flow:   (n_dates, n_leads)
    """  

    basin_id = f"{int(GAGE_ID):08d}"

    basin_dir = os.path.join(
        config["sim_dir"],
        "GEFS_1year_forecasts",
        f"{START_DATE}_{END_DATE}_pre-{PRE}_w{WINDOW}_post-{POST}",
        basin_id,
    )

    files = sorted(glob.glob(
        os.path.join(basin_dir, "forecast_*.npz")
    ))

    if not files:
        raise FileNotFoundError(
            f"No forecast files found in {basin_dir}"
        )

    forecasts = []
    init_dates = []

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    for path in files:
        with np.load(path) as data:
            fc = np.asarray(data["forecast"], dtype=float)
            init_date = pd.Timestamp(
                str(data["init_date"].item())
            ).normalize()

        if not (start <= init_date <= end):
            continue

        if fc.shape != (N_ENSEMBLES, N_LEADS):
            raise ValueError(
                f"Unexpected forecast shape {fc.shape}: {path}"
            )

        forecasts.append(fc.T)
        init_dates.append(init_date)

    if not forecasts:
        raise ValueError("No forecasts in the requested date range")

    forecasts = np.stack(forecasts)
    init_dates = pd.DatetimeIndex(init_dates)

    if not init_dates.is_unique:
        raise ValueError(f"Duplicate initialization dates: {basin_id}")

    # Find the basin in existing calibration diagnostics
    basin_data = next(
        (
            value for key, value in basin_results.items()
            if int(key) == int(GAGE_ID)
        ),
        None,
    )

    if basin_data is None:
        raise KeyError(f"Basin {basin_id} not found in basin_results")

    reference = REFERENCE.lower()

    if reference == "observed":
        data_key = "obs"
    elif reference == "simulated":
        data_key = "sim"
    else:
        raise ValueError("REFERENCE must be 'observed' or 'simulated'")

    qref = np.asarray(basin_data[data_key], dtype=float).reshape(-1)
    qref_dates = pd.DatetimeIndex(
        basin_data["dates"]
    ).normalize()

    if len(qref) != len(qref_dates):
        raise ValueError("Reference dates and values have different lengths")

    if not qref_dates.is_unique:
        raise ValueError("Duplicate reference dates")

    qref_series = pd.Series(qref, index=qref_dates)

    offsets = np.arange(
        LEAD1_OFFSET,
        LEAD1_OFFSET + N_LEADS
    )

    valid_dates = (
        init_dates.values[:, None]
        + offsets[None, :].astype("timedelta64[D]")
    )

    target_flow = qref_series.reindex(
        pd.DatetimeIndex(valid_dates.ravel())
    ).to_numpy().reshape(valid_dates.shape)

    ensemble_mean = np.mean(forecasts, axis=2)

    return ensemble_mean, target_flow


def calculate_gefs_ns(ensemble_mean, target_flow, MIN_VALID_CASES=30):
    """Nash-Sutcliffe efficiency for each forecast lead.""" 

    n_leads = ensemble_mean.shape[1]
    ns = np.full(n_leads, np.nan)

    for j in range(n_leads):
        valid = (
            np.isfinite(ensemble_mean[:, j])
            & np.isfinite(target_flow[:, j])
        )

        if valid.sum() < MIN_VALID_CASES:
            continue

        pred = ensemble_mean[valid, j]
        obs = target_flow[valid, j]

        denominator = np.sum((obs - np.mean(obs)) ** 2)

        if denominator > 0:
            ns[j] = 1 - np.sum((pred - obs) ** 2) / denominator

    return ns


def calculate_gefs_pbias(ensemble_mean, target_flow, MIN_VALID_CASES=30):
    """
    Percent bias following Alfieri et al. (2014).
    Positive = underprediction; negative = overprediction.
    """ 

    n_leads = ensemble_mean.shape[1]
    pbias = np.full(n_leads, np.nan)

    for j in range(n_leads):
        valid = (
            np.isfinite(ensemble_mean[:, j])
            & np.isfinite(target_flow[:, j])
        )

        if valid.sum() < MIN_VALID_CASES:
            continue

        pred = ensemble_mean[valid, j]
        obs = target_flow[valid, j]

        if not np.isclose(np.mean(obs), 0.0, atol=1e-12):
            pbias[j] = 100 * (
                np.mean(obs) - np.mean(pred)
            ) / np.mean(obs)

    return pbias


def calculate_gefs_cv(ensemble_mean, target_flow, MIN_VALID_CASES=30):
    """Coefficient of variation of RMSE for each forecast lead.""" 

    n_leads = ensemble_mean.shape[1]
    cv = np.full(n_leads, np.nan)

    for j in range(n_leads):
        valid = (
            np.isfinite(ensemble_mean[:, j])
            & np.isfinite(target_flow[:, j])
        )

        if valid.sum() < MIN_VALID_CASES:
            continue

        pred = ensemble_mean[valid, j]
        obs = target_flow[valid, j]

        mean_obs = np.mean(obs)

        if not np.isclose(mean_obs, 0.0, atol=1e-12):
            rmse = np.sqrt(np.mean((pred - obs) ** 2))
            cv[j] = rmse / mean_obs

    return cv

def plot_gefs_metric(
    values,
    metric_name,
    save_dir=None,
    START_DATE=None,
    END_DATE=None,
    basin_id=None,
    percentiles=None,
    ): 

    values = np.asarray(values, dtype=float)
    leads = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Same line style as CRPSS
    ax.plot(
        leads,
        values,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label=metric_name,
        zorder=3,
    )

    # Optional spatial percentile bands
    if percentiles is not None:
        p05, p25, p75, p95 = [
            np.asarray(p, dtype=float) for p in percentiles
        ]

        ax.fill_between(
            leads, p05, p95,
            color="lightgray",
            alpha=0.75,
            label="5–95%",
            zorder=1,
        )

        ax.fill_between(
            leads, p25, p75,
            color="gray",
            alpha=0.65,
            label="25–75%",
            zorder=2,
        )

    # Reference line
    ax.axhline(
        0,
        color="gray",
        linewidth=1,
        zorder=0,
    )

    ax.set_xlim(0.5, len(leads) + 0.5)
    ax.set_xticks(leads)

    ax.set_xlabel("Lead time [days]", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)

    title = f"GEFS Forecast {metric_name} Against Observed"

    if basin_id is not None:
        title += f"\nBasin {int(basin_id):08d}"

    if START_DATE is not None and END_DATE is not None:
        title += f" ({START_DATE} to {END_DATE})"

    ax.set_title(title, fontsize=15)

    ax.tick_params(axis="both", labelsize=11)

    if percentiles is not None:
        ax.legend(frameon=False, loc="best")

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        safe_name = (
            metric_name.replace(" ", "_")
                       .replace("/", "_")
                       .replace("%", "pct")
        )

        path = os.path.join(
            save_dir,
            f"GEFS_{safe_name}.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print("Saved:", path)

    plt.show()
    plt.close(fig)
    
    


def print_gefs_summary(target_basins, gefs_results, failed_basins):
    """
    Print a summary of GEFS forecast processing.

    Parameters
    ----------
    target_basins : list
        Requested basin IDs.
    gefs_results : dict
        Successful basin-level function results.
    failed_basins : dict
        Basin IDs with failed dates or execution errors.
    """

    print("\n========== FORECAST SUMMARY ==========", flush=True)

    print("Requested basins:", len(target_basins), flush=True)
    print("Completed basin calls:", len(gefs_results), flush=True)
    print("Basins with failures:", len(failed_basins), flush=True)

    print(
        "New forecasts:",
        sum(r["n_saved"] for r in gefs_results.values()),
        flush=True
    )

    print(
        "Skipped forecasts:",
        sum(r["n_skipped"] for r in gefs_results.values()),
        flush=True
    )

    print(
        "Failed forecast dates:",
        sum(r["n_failed"] for r in gefs_results.values()),
        flush=True
    )

    print("\nFailed basins:", failed_basins, flush=True)
    print("End time:", datetime.now(), flush=True)
    
 
def build_observed_basin_results(
    data_loader,
    config,
    loader_basin_pool,
    selected_basins,
    WARMUP=365,
):
    """
    Build observed reference data for GEFS CRPSS evaluation.

    No HBV simulation or calibration diagnostics required.
    """

    # 1. Load observed discharge
    target = data_loader.dataset["target"]
    n_timesteps = target.shape[0]

    # 2. Construct simulation dates
    dates_all = get_simulation_dates(config, n_timesteps)
    dates = dates_all[WARMUP:]

    # 3. Basin ID to dataset index mapping
    basin_lookup = {
        int(bid): idx
        for idx, bid in enumerate(loader_basin_pool)
    }

    if target.shape[1] != len(loader_basin_pool):
        raise ValueError("Basin ordering/count mismatch")

    # 4. Build reference dictionary
    basin_results = {}

    for gage_id in selected_basins:

        gage_id = int(gage_id)

        if gage_id not in basin_lookup:
            raise ValueError(
                f"Basin {gage_id:08d} not found in data_loader"
            )

        basin_idx = basin_lookup[gage_id]

        obs = target[WARMUP:, basin_idx, 0]

        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        obs = np.asarray(obs).reshape(-1)

        if len(obs) != len(dates):
            raise ValueError(
                f"Date alignment error for {gage_id:08d}"
            )

        basin_results[f"{gage_id:08d}"] = {
            "dates": dates,
            "obs": obs,
        }

    return basin_results