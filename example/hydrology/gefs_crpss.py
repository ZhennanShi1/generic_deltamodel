"""GEFS streamflow forecast storage, verification, and CRPSS evaluation.

The CRPSS/evaluation functions are independent of tempfunctions.py.
Forecast generation accepts the hydrological model's functions as callbacks.
The saved NPZ layout and Alfieri-style score definitions are preserved.

Forecast NPZ: forecast shape (n_members, n_leads), init_date, basin_id.
Loaded forecasts: (n_init_dates, n_leads, n_members).
GEFS convention: lead1_offset=0 means Day 1 verifies ON initialization date.
The persistence forecast uses the preceding day's discharge.
For exact comparability, both reference CRPSS scores use the SAME valid cases;
thus missing pre-initialization flow also excludes a climatology score case.
Climatology_years=None uses the complete provided reference series, which can
include verification-period information. Set years to historical training years
to avoid look-ahead bias; this changes the reference and therefore the CRPSS.

CRPS references: Alfieri et al. (2014), Eqs. (4), (5), (7), (8).
Ensemble kernel CRPS: Gneiting & Raftery (2007), nonnegative form of Eq. (21).
"""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd

# ===========================================================================
# 1. Forecast paths and NPZ I/O
# ===========================================================================

def forecast_root(config, START_DATE="2012-01-01", END_DATE="2012-12-31",
                  PRE="scalar", WINDOW=30, POST=True):
    """Get folder for an experiment without reading or writing forecasts.

    Args:
        config: dict with config['sim_dir'] (str or Path).
        START_DATE, END_DATE: date strings; PRE: str; WINDOW: int; POST: bool.
    Returns: pathlib.Path, directory (not necessarily existing).
    """
    return Path(config["sim_dir"]) / "GEFS_1year_forecasts" / (
        f"{START_DATE}_{END_DATE}_pre-{PRE}_w{WINDOW}_post-{POST}"
    )


def forecast_file_path(save_root, basin_id, init_date):
    """Path to a single NPZ file.

    Args: save_root (Path/str); basin_id (str/int); init_date (date-like scalar).
    Returns: Path, ending in <basin>/forecast_YYYYMMDD.npz.
    """
    date = pd.Timestamp(init_date).normalize()
    return Path(save_root) / f"{int(basin_id):08d}" / f"forecast_{date:%Y%m%d}.npz"


def save_forecast_npz(save_root, basin_id, init_date, forecast, overwrite=True):
    """Store one ensemble forecast atomically; does NOT compute CRPS.

    Args:
        save_root: Path/str, experiment directory.
        basin_id: scalar basin identifier.
        init_date: scalar date of initialization.
        forecast: real ndarray (M,L); M=members (5), L=leads (10).
        overwrite: bool, replace existing file if True.
    NPZ keys: forecast (M,L), init_date (str), basin_id (str).
    Returns: Path to saved NPZ.
    """
    values = np.asarray(forecast)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("forecast must contain real numeric values")
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("forecast must have shape (n_members, n_leads)")

    path = forecast_file_path(save_root, basin_id, init_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=path.parent, delete=False
        ) as out:
            temporary = Path(out.name)
            np.savez_compressed(
                out,
                forecast=values,
                init_date=pd.Timestamp(init_date).strftime("%Y-%m-%d"),
                basin_id=f"{int(basin_id):08d}",
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_forecast_npz(path, n_members=None, n_leads=None):
    """Read exactly one saved ensemble forecast.

    Args: path (Path/str); n_members, n_leads (optional ints).
    Returns: (forecast ndarray (M,L), init_date Timestamp, basin_id str|None).
    Supports legacy NPZ files without basin_id.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        forecast = np.asarray(data["forecast"], dtype=float)
        init_date = pd.Timestamp(str(data["init_date"].item())).normalize()
        basin_id = str(data["basin_id"].item()) if "basin_id" in data else None

    if forecast.ndim != 2 or 0 in forecast.shape:
        raise ValueError(f"Invalid forecast dimensions {forecast.shape}: {path}")
    if n_members is not None and forecast.shape[0] != n_members:
        raise ValueError(f"Expected {n_members} members, got {forecast.shape}: {path}")
    if n_leads is not None and forecast.shape[1] != n_leads:
        raise ValueError(f"Expected {n_leads} leads, got {forecast.shape}: {path}")
    return forecast, init_date, basin_id


def load_basin_forecasts(save_root, basin_id, *, year=None, start_date=None,
                         end_date=None, n_members=None, n_leads=None):
    """Load all matching dates for one basin into one array.

    Args:
        save_root: Path/str; basin_id: str/int; year: optional int.
        start_date/end_date: optional inclusive date limits.
        n_members/n_leads: optional expected sizes (M,L).
    Returns:
        forecasts: ndarray (I,L,M), sorted by initialization date.
        init_dates: DatetimeIndex (I,).
    Example I=362, L=10, M=5; missing forecast files reduce I.
    """
    folder = Path(save_root) / f"{int(basin_id):08d}"
    paths = sorted(folder.glob("forecast_*.npz"))
    start = pd.Timestamp(start_date).normalize() if start_date is not None else None
    end = pd.Timestamp(end_date).normalize() if end_date is not None else None
    rows = []

    for path in paths:
        fc, date, saved_basin = load_forecast_npz(path, n_members, n_leads)
        expected_name = f"forecast_{date:%Y%m%d}.npz"
        if path.name != expected_name:
            raise ValueError(f"Forecast filename and init_date disagree: {path}")
        if saved_basin is not None and int(saved_basin) != int(basin_id):
            raise ValueError(f"Basin ID in {path} does not match requested basin")
        if year is not None and date.year != int(year):
            continue
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        rows.append((date, fc.T))

    if not rows:
        raise FileNotFoundError(f"No matching forecast NPZs in {folder}")
    rows.sort(key=lambda item: item[0])
    init_dates = pd.DatetimeIndex([item[0] for item in rows])
    if not init_dates.is_unique:
        raise ValueError(f"Duplicate forecast initialization dates for {basin_id}")
    forecasts = np.stack([item[1] for item in rows])
    return forecasts, init_dates


# ===========================================================================
# 2. Reference flow and date alignment
# ===========================================================================

def get_simulation_dates(config, n_timesteps):
    """Construct the daily date axis used by data_loader.dataset.

    Args: config['sim_time'] with starting date; n_timesteps: positive int.
    Returns: DatetimeIndex (T,). Important: validate loader date alignment.
    """
    sim_time = config["sim_time"]
    if isinstance(sim_time, (list, tuple)):
        start = sim_time[0]
    elif isinstance(sim_time, dict):
        start = next((sim_time[k] for k in (
            "start", "start_date", "start_time", "begin"
        ) if k in sim_time), None)
        if start is None:
            raise ValueError(f"Unknown sim_time format: {sim_time}")
    elif isinstance(sim_time, (str, pd.Timestamp)):
        start = sim_time
    else:
        raise ValueError(f"Unsupported sim_time format: {sim_time}")
    return pd.date_range(start=pd.to_datetime(start), periods=n_timesteps, freq="D")


def build_observed_basin_results(data_loader, config, loader_basin_pool,
                                 selected_basins, WARMUP=365):
    """Extract daily observed discharge after warm-up for requested basins.

    Args:
        data_loader.dataset['target']: Tensor/ndarray (T,B,>=1).
        config: dict with 'sim_time'; loader_basin_pool: IDs (B,) in loader order.
        selected_basins: IDs (K,); WARMUP: number of timesteps omitted.
    Returns: dict[8_digit_id]={'dates': DatetimeIndex(T-WARMUP,),
                                'obs': ndarray(T-WARMUP,)}.
    This function does not simulate discharge; reference='simulated' requires
    caller-provided 'sim' arrays of the same length.
    """
    target = data_loader.dataset["target"]
    if target.shape[1] != len(loader_basin_pool):
        raise ValueError("Basin ordering/count mismatch")
    dates = get_simulation_dates(config, target.shape[0])[WARMUP:]
    lookup = {int(bid): idx for idx, bid in enumerate(loader_basin_pool)}
    result = {}
    for bid in selected_basins:
        bid = int(bid)
        if bid not in lookup:
            raise KeyError(f"Basin {bid:08d} not found in loader_basin_pool")
        obs = target[WARMUP:, lookup[bid], 0]
        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy()
        obs = np.asarray(obs, dtype=float).reshape(-1)
        if len(obs) != len(dates):
            raise ValueError(f"Date alignment error for basin {bid:08d}")
        result[f"{bid:08d}"] = {"dates": dates, "obs": obs}
    return result


def reference_series(basin_results, basin_id, reference="observed"):
    """Turn one basin's reference discharge into a date-indexed series.

    Args: basin_results: mapping of {id: {'dates':(T,), 'obs':(T,),
          optionally 'sim':(T,)}}; basin_id: str/int; reference: observed/simulated.
    Returns: pandas.Series (T,) indexed by unique normalized dates.
    """
    data = next((v for k, v in basin_results.items()
                 if int(k) == int(basin_id)), None)
    if data is None:
        raise KeyError(f"Missing basin_results for {int(basin_id):08d}")
    if reference not in ("observed", "simulated"):
        raise ValueError("reference must be 'observed' or 'simulated'")
    key = "obs" if reference == "observed" else "sim"
    values = np.asarray(data[key], dtype=float).reshape(-1)
    dates = pd.DatetimeIndex(pd.to_datetime(data["dates"])).normalize()
    if len(dates) != len(values) or not dates.is_unique:
        raise ValueError(f"Reference dates are misaligned for {basin_id}")
    return pd.Series(values, index=dates).sort_index()


def align_reference(qref_series, init_dates, n_leads, lead1_offset=0):
    """Match forecast valid times and persistence time to reference series.

    Args:
        qref_series: pandas.Series (T,) indexed by daily dates.
        init_dates: DatetimeIndex (I,); n_leads: int L.
        lead1_offset: int; 0 => Day1=init, 1 => Day1=init+1.
    Returns:
        target_flow: ndarray (I,L), observed flow at init+offset+lead_index.
        initial_flow: ndarray (I,), observed flow at init+offset-1.
    Missing dates are NaN; they are NOT filled or interpolated.
    """
    if not isinstance(qref_series, pd.Series):
        raise TypeError("qref_series must be a pandas Series indexed by dates")
    if not qref_series.index.is_unique:
        raise ValueError("Reference index contains duplicate dates")
    if n_leads < 1 or not isinstance(lead1_offset, (int, np.integer)):
        raise ValueError("n_leads must be positive and lead1_offset an integer")
    init_dates = pd.DatetimeIndex(pd.to_datetime(init_dates)).normalize()
    if not init_dates.is_unique:
        raise ValueError("Initialization dates contain duplicates")
    offsets = np.arange(lead1_offset, lead1_offset + n_leads)
    valid_dates = (
        init_dates.values[:, None] + offsets[None, :].astype("timedelta64[D]")
    )
    targets = qref_series.reindex(
        pd.DatetimeIndex(valid_dates.ravel())
    ).to_numpy().reshape(len(init_dates), n_leads)
    persistence_dates = init_dates + pd.Timedelta(days=lead1_offset - 1)
    initial = qref_series.reindex(persistence_dates).to_numpy()
    return targets, initial


def climatology_values(qref_series, years=None):
    """Select observed/simulated historical discharge sample.

    Args: qref_series pandas.Series (T,), years iterable of years or None.
    Returns: finite ndarray (C,), used by Alfieri reference Eq.7.
    CAUTION: years=None includes all available dates, potentially the test year.
    """
    if years is None:
        values = qref_series.to_numpy()
    else:
        values = qref_series.loc[qref_series.index.year.isin(list(years))].to_numpy()
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No valid climatological reference data")
    return values


def valid_forecast_cases(forecasts, target_flow, initial_flow):
    """Shared valid-case mask for fair persistence-vs-climatology comparison.

    Args: forecasts ndarray (I,L,M), target_flow ndarray (I,L),
          initial_flow ndarray (I,).
    Returns: boolean ndarray (I,L), True only if all members, target, and
    pre-initialization persistence are finite.
    """
    fc = np.asarray(forecasts, dtype=float)
    target = np.asarray(target_flow, dtype=float)
    initial = np.asarray(initial_flow, dtype=float)
    if fc.ndim != 3 or target.shape != fc.shape[:2] or initial.shape != (fc.shape[0],):
        raise ValueError("Expected forecasts (I,L,M), target (I,L), initial (I,)")
    return (np.isfinite(fc).all(axis=2)
            & np.isfinite(target)
            & np.isfinite(initial[:, None]))


# ===========================================================================
# 3. Scientific score equations (small and independently testable)
# ===========================================================================

def crps_single(esp, obs):
    """Kernel CRPS for exactly one probabilistic forecast case.

    Args: esp: real 1D ndarray (M,), one member value each; obs: float.
    Returns: scalar CRPS (float), NaN if observation/member missing.
    Formula (Gneiting & Raftery 2007, positive-loss Eq.21):
        mean(|esp - obs|) - 1/2 * mean(|esp_i - esp_j|).
    """
    esp = np.asarray(esp, dtype=float)
    if esp.ndim != 1 or esp.size == 0:
        raise ValueError("esp must have nonempty shape (n_members,)")
    if not np.isfinite(obs) or not np.isfinite(esp).all():
        return np.nan
    term1 = np.mean(np.abs(esp - obs))
    term2 = 0.5 * np.mean(np.abs(esp[:, None] - esp[None, :]))
    return float(term1 - term2)


def ensemble_crps_cases(forecasts, target_flow):
    """Use crps_single once per initialization and lead (no duplicate equation).

    Args: forecasts ndarray (I,L,M); target_flow ndarray (I,L).
    Returns: per-case CRPS ndarray (I,L); missing cases -> NaN.
    """
    fc = np.asarray(forecasts, dtype=float)
    obs = np.asarray(target_flow, dtype=float)

    if fc.ndim != 3 or 0 in fc.shape or obs.shape != fc.shape[:2]:
        raise ValueError("Expected nonempty forecasts (n_init,n_leads,n_members) "
                         "and target_flow (n_init,n_leads)")

    n_init, n_leads, _ = fc.shape

    crps_cases = np.full((n_init, n_leads), np.nan)

    for i in range(n_init):
        for j in range(n_leads):
            crps_cases[i, j] = crps_single(
                esp=fc[i, j, :],
                obs=obs[i, j]
            )

    return crps_cases

def average_by_lead(case_scores, valid):
    """Average any per-case score over MATCHED valid initializations.

    Args: case_scores ndarray (I,L); valid boolean ndarray (I,L).
    Returns: mean score ndarray (L,), NaN for leads with no valid cases.
    """
    scores = np.asarray(case_scores, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if scores.ndim != 2 or valid.shape != scores.shape:
        raise ValueError("Scores and valid must both be (n_init, n_leads)")
    result = np.full(scores.shape[1], np.nan)
    for j in range(scores.shape[1]):
        if np.any(valid[:, j]):
            result[j] = np.mean(scores[valid[:, j], j])
    return result


def average_crps(forecasts, target_flow, valid):
    """Calculate forecast CRPS by lead from ensemble forecasts.

    Args: forecasts (I,L,M); target_flow (I,L); valid bool (I,L).
    Returns: ndarray (L,) of mean CRPS over valid initialization dates.
    """
    return average_by_lead(ensemble_crps_cases(forecasts, target_flow), valid)


def crps_ref_climatology(qsim):
    """Alfieri (2014) Eq.7: deterministic mean-discharge reference.

    Args: qsim 1D ndarray (C,), finite historical daily discharge sample.
    Returns: scalar mean(|qsim(t) - mean(qsim)|) over C days.
    This is a constant scalar across lead times, NOT an ensemble CDF.
    """
    qsim = np.asarray(qsim, dtype=float)
    qsim = qsim[np.isfinite(qsim)]
    if not qsim.size:
        return np.nan
    return float(np.mean(np.abs(qsim - np.mean(qsim))))


def crps_ref_persistence(target_flow, initial_flow, valid):
    """Alfieri (2014) Eq.8: persistence MAE for each lead.

    Args: target_flow ndarray (I,L), initial_flow ndarray (I,),
          valid boolean ndarray (I,L).
    Returns: ndarray (L,) of mean(|target - initial|) per lead.
    """
    errors = np.abs(np.asarray(target_flow) - np.asarray(initial_flow)[:, None])
    return average_by_lead(errors, valid)


def calculate_crpss(crps_fc, crps_ref, n_valid, min_valid_cases=30):
    """Alfieri (2014) Eq.4: 1 - CRPS_forecast/CRPS_reference.

    Args: crps_fc, crps_ref arrays broadcastable to shape (L,),
          n_valid ndarray (L,) or scalar; min_valid_cases: int.
    Returns: ndarray (L,), NaN when insufficient cases or denominator <=0.
    """
    fc, ref, n = np.broadcast_arrays(
        np.asarray(crps_fc, dtype=float),
        np.asarray(crps_ref, dtype=float),
        np.asarray(n_valid),
    )
    result = np.full(fc.shape, np.nan)
    valid = ((n >= min_valid_cases) & np.isfinite(fc)
             & np.isfinite(ref) & (ref > 0))
    result[valid] = 1 - fc[valid] / ref[valid]
    return result


# ===========================================================================
# 4. One basin: load -> align -> calculate -> return
# ===========================================================================

def calculate_basin_crpss(forecasts, init_dates, qref_series, *,
                         lead1_offset=0, min_valid_cases=30,
                         climatology_years=None):
    """Compute all lead-wise CRPS/CRPSS for one basin from loaded data.

    Args:
        forecasts: ndarray (I,L,M).
        init_dates: DatetimeIndex (I,).
        qref_series: pandas.Series (T,) indexed by daily reference dates.
        lead1_offset: 0 => verify Day1 on init date.
        min_valid_cases: int; climatology_years: iterable or None.
    Returns: dict with score arrays (L,), n_valid (L,), and metadata.
    Forecast and both refs use the SAME mask; historical climatology uses
    all finite dates selected by climatology_years (not I forecast cases).
    """
    fc = np.asarray(forecasts, dtype=float)
    if fc.ndim != 3 or 0 in fc.shape or len(init_dates) != fc.shape[0]:
        raise ValueError("forecasts must be nonempty (n_init,n_leads,n_members)")
    n_leads = fc.shape[1]

    target, initial = align_reference(qref_series, init_dates, n_leads, lead1_offset)
    valid = valid_forecast_cases(fc, target, initial)
    n_valid = valid.sum(axis=0)
    clim = climatology_values(qref_series, climatology_years)

    crps_fc = average_crps(fc, target, valid)
    crps_pf = crps_ref_persistence(target, initial, valid)
    clim_crps = crps_ref_climatology(clim)
    crps_ad = np.where(n_valid > 0, clim_crps, np.nan)

    return {
        "CRPS_forecast": crps_fc,
        "CRPS_ref_pf": crps_pf,
        "CRPS_ref_ad": crps_ad,
        "CRPSS_pf": calculate_crpss(crps_fc, crps_pf, n_valid, min_valid_cases),
        "CRPSS_ad": calculate_crpss(crps_fc, crps_ad, n_valid, min_valid_cases),
        "n_valid": n_valid,
        "n_init_dates": len(init_dates),
        "climatology_mean": float(np.mean(clim)),
        "n_climatology_days": len(clim),
    }


def evaluate_gefs_basin(basin_id, basin_results, save_root, *,
                        year=2012, n_leads=10, n_ensembles=5,
                        lead1_offset=0, min_valid_cases=30,
                        reference="observed", climatology_years=None,
                        start_date=None, end_date=None):
    """Read saved forecasts and evaluate one basin; no HBV run.

    Args: basin_id int/str; basin_results mapping; save_root Path/str;
          year int|None, start_date/end_date optional dates, n_leads L,
          n_ensembles M, other CRPSS options as in calculate_basin_crpss.
    Returns: dict with each score ndarray (L,), basin_id, metadata.
    """
    fc, dates = load_basin_forecasts(
        save_root, basin_id, year=year, start_date=start_date, end_date=end_date,
        n_members=n_ensembles, n_leads=n_leads
    )
    qref = reference_series(basin_results, basin_id, reference)
    result = calculate_basin_crpss(
        fc, dates, qref, lead1_offset=lead1_offset,
        min_valid_cases=min_valid_cases, climatology_years=climatology_years,
    )
    return {"basin_id": f"{int(basin_id):08d}", "reference": reference, **result}


# ===========================================================================
# 5. Summary, CSV, and figures
# ===========================================================================

def basin_scores_dataframe(basin_scores, reference):
    """Flatten basin score dictionaries to one row per basin and lead.

    Args: basin_scores list of K dicts with score ndarray (L,);
          reference str.
    Returns: DataFrame with K*L rows.
    """
    rows = []
    for result in basin_scores:
        for j in range(len(result["CRPSS_pf"])):
            rows.append({
                "basin_id": result["basin_id"],
                "reference": reference,
                "lead_time": j + 1,
                **{name: result[name][j] for name in (
                    "CRPSS_pf", "CRPSS_ad", "CRPS_forecast",
                    "CRPS_ref_pf", "CRPS_ref_ad", "n_valid"
                )},
            })
    return pd.DataFrame(rows)


def spatial_summary(basin_scores):
    """Calculate spatial mean and percentile bands by lead.

    Args: basin_scores list of K dicts, each score ndarray (L,).
    Returns: DataFrame L rows, across-basin mean/p05/p25/p75/p95.
    """
    pf = np.stack([b["CRPSS_pf"] for b in basin_scores])
    ad = np.stack([b["CRPSS_ad"] for b in basin_scores])

    def stat(a, pct=None):
        out = []
        for col in a.T:
            finite = col[np.isfinite(col)]
            out.append(np.nan if not finite.size else (
                finite.mean() if pct is None else np.percentile(finite, pct)
            ))
        return out

    return pd.DataFrame({
        "lead_time": np.arange(1, pf.shape[1] + 1),
        "CRPSS_pf_mean": stat(pf),
        "CRPSS_ad_mean": stat(ad),
        "CRPSS_pf_p05": stat(pf, 5),
        "CRPSS_pf_p25": stat(pf, 25),
        "CRPSS_pf_p75": stat(pf, 75),
        "CRPSS_pf_p95": stat(pf, 95),
        "CRPSS_ad_p25": stat(ad, 25),
        "CRPSS_ad_p75": stat(ad, 75),
        "n_basins_pf": np.isfinite(pf).sum(axis=0),
        "n_basins_ad": np.isfinite(ad).sum(axis=0),
    })


def plot_crpss(summary_df, *, mode, n_basins, reference,
               start_date, end_date, save_path, show_plot=True):
    """Plot CRPSS versus lead time and write a figure.

    Args: summary_df DataFrame (L rows), mode str, n_basins int,
          reference str, start/end_date str, save_path Path/str, show_plot bool.
    Returns: Path to saved PNG. Negative CRPSS is not clipped.
    """
    import matplotlib.pyplot as plt

    leads = summary_df["lead_time"].to_numpy()
    def vals(name):
        return summary_df[name].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    if mode == "multiple" and n_basins > 1:
        ax.fill_between(leads, vals("CRPSS_pf_p05"), vals("CRPSS_pf_p95"),
                        color="lightgray", alpha=.8, label="5–95% (persistence)")
        ax.fill_between(leads, vals("CRPSS_pf_p25"), vals("CRPSS_pf_p75"),
                        color="gray", alpha=.65, label="25–75% (persistence)")
        ax.plot(leads, vals("CRPSS_ad_p25"), "k:", linewidth=1)
        ax.plot(leads, vals("CRPSS_ad_p75"), "k:", linewidth=1,
                label="25–75% (average discharge)")
    ax.plot(leads, vals("CRPSS_pf_mean"), "k-", linewidth=2, label=r"CRPSS$_{pf}$")
    ax.plot(leads, vals("CRPSS_ad_mean"), "k--", linewidth=2, label=r"CRPSS$_{ad}$")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xlabel("Lead time [days]")
    ax.set_ylabel("CRPSS")
    ax.set_xticks(leads)
    ax.set_title(f"GEFS Forecast Skill Against {reference.title()}\n"
                 f"Across {n_basins} Basin(s) ({start_date} to {end_date})")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return save_path


def run_gefs_crpss(mode, config, basin_results, basin_pool=None, GAGE_ID=None,
                  START_DATE="2012-01-01", END_DATE="2012-12-31",
                  PRE="scalar", WINDOW=30, POST=True,
                  YEAR=2012, N_LEADS=10, N_ENSEMBLES=5,
                  LEAD1_OFFSET=0, MIN_VALID_CASES=30, CLIMATOLOGY_YEARS=None,
                  show_plot=True):
    """Orchestrate CRPSS from existing NPZ files, across one/many basins.

    Args: mode 'single'|'multiple'; config dict; basin_results dict;
          basin_pool list of K IDs for multiple mode; GAGE_ID single ID;
          START_DATE/END_DATE str; PRE str, WINDOW int, POST bool; YEAR int;
          N_LEADS=L, N_ENSEMBLES=M; LEAD1_OFFSET int;
          MIN_VALID_CASES int; REFERENCE observed/simulated;
          CLIMATOLOGY_YEARS iterable[int] or None; show_plot bool.
    Returns: dict with basin_scores K, arrays (K,L), results_df (K*L rows),
    summary_df (L rows), errors, CSV/PNG paths. Does not rerun HBV.
    """
    mode = mode.lower().strip()
    reference = REFERENCE.lower().strip()
    if mode == "single":
        if GAGE_ID is None:
            raise ValueError("GAGE_ID is required when mode='single'")
        selected = [int(GAGE_ID)]
    elif mode == "multiple":
        if basin_pool is None:
            raise ValueError("basin_pool is required when mode='multiple'")
        selected = list(dict.fromkeys(int(b) for b in basin_pool))
    else:
        raise ValueError("mode must be 'single' or 'multiple'")
    if not selected:
        raise ValueError("No basins selected")
    if reference not in ("observed", "simulated"):
        raise ValueError("REFERENCE must be 'observed' or 'simulated'")

    root = forecast_root(config, START_DATE, END_DATE, PRE, WINDOW, POST)
    output_dir = root / f"CRPSS_{reference}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    basin_scores, failures = [], []
    for i, basin_id in enumerate(selected):
        try:
            score = evaluate_gefs_basin(
                basin_id, basin_results, root, year=YEAR,
                n_leads=N_LEADS, n_ensembles=N_ENSEMBLES,
                lead1_offset=LEAD1_OFFSET, min_valid_cases=MIN_VALID_CASES,
                reference=reference, climatology_years=CLIMATOLOGY_YEARS,
                start_date=START_DATE, end_date=END_DATE,
            )
            basin_scores.append(score)
            print(f"[{i+1}/{len(selected)}] {basin_id:08d}: "
                  f"{score['n_init_dates']} forecasts")
        except Exception as exc:
            failures.append({"basin_id": f"{basin_id:08d}", "error": str(exc)})
            print(f"FAILED {basin_id:08d}: {exc}")
    if not basin_scores:
        raise RuntimeError("No basins evaluated successfully")

    results_df = basin_scores_dataframe(basin_scores, reference)
    summary_df = spatial_summary(basin_scores)
    csv_path = output_dir / f"CRPSS_{mode}_{reference}_{YEAR}.csv"
    summary_csv = output_dir / f"CRPSS_summary_{mode}_{reference}_{YEAR}.csv"
    results_df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pd.DataFrame(failures, columns=["basin_id", "error"]).to_csv(
        output_dir / "failed_basins.csv", index=False
    )
    fig_name = (f"CRPSS_{basin_scores[0]['basin_id']}_{reference}_{YEAR}.png"
                if mode == "single" else f"CRPSS_spatial_{reference}_{YEAR}.png")
    fig_path = plot_crpss(
        summary_df, mode=mode, n_basins=len(basin_scores), reference=reference,
        start_date=START_DATE, end_date=END_DATE,
        save_path=output_dir / fig_name, show_plot=show_plot,
    )
    print(f"Evaluated: {len(basin_scores)}; failed: {len(failures)}")
    print(f"Figure: {fig_path}\nCSV: {csv_path}")
    return {
        "mode": mode,
        "reference": reference,
        "basin_crpss": basin_scores,
        "crpss_pf_all": np.stack([b["CRPSS_pf"] for b in basin_scores]),
        "crpss_ad_all": np.stack([b["CRPSS_ad"] for b in basin_scores]),
        "results_df": results_df,
        "summary_df": summary_df,
        "failed_basins": failures,
        "figure_path": str(fig_path),
        "csv_path": str(csv_path),
        "summary_csv": str(summary_csv),
        "output_dir": str(output_dir),
    }


# ===========================================================================
# 6. Forecast generation (same external model callbacks as the old function)
# ===========================================================================

def _forecast_one_date(*, basin_id, basin_idx, pred_date, forecast_days,
                       n_ensembles, warmup, window, pre, post,
                       model, data_loader, config, device, gefs_dir, state_path,
                       startid_endid, run_segment, run_warm_forecasts_restart,
                       post_processing):
    """Run HBV for one basin, one initialization date and all members.

    Args: basin_id/basin_idx scalar; pred_date date; forecast_days L;
          n_ensembles M; model, loader, config, device and GEFS directories;
          startid_endid, run_segment, run_warm_forecasts_restart and
          post_processing are tempfunctions callbacks.
    Returns: ndarray (M,L) of forecast streamflow.
    """
    sidx, _, timesteps, _ = startid_endid(config, warmup, pred_date, forecast_days)
    name = next(iter(model.model_dict))
    history = run_segment(
        model=model, dataset=data_loader.dataset, basin_idx=basin_idx,
        device=device, name=name, start_idx=None, end_idx=sidx,
        warm_up=warmup, cache_states=True, warm_up_states=True,
    )
    model.verbose = False
    model.save_states()
    ensemble = run_warm_forecasts_restart(
        model=model, data_loader=data_loader, state_path=state_path,
        gage_id=basin_id, basin_idx=basin_idx, start_date=pred_date,
        horizon=forecast_days, N_ENSEMBLES=n_ensembles, GEFS_DIR=gefs_dir,
        timesteps=timesteps, WINDOW=window, CORRECTION=pre,
        device=device, verbose=False,
    )
    if post:
        ensemble = post_processing(history, ensemble)
    if hasattr(ensemble, "detach"):
        ensemble = ensemble.detach().cpu().numpy()
    ensemble = np.asarray(ensemble)
    if ensemble.shape != (n_ensembles, forecast_days):
        raise ValueError(f"Unexpected forecast shape {ensemble.shape}")
    return ensemble


def run_gefs_one_year(GAGE_ID, model, data_loader, config, basin_pool,
                      startid_endid, run_segment, run_warm_forecasts_restart,
                      post_processing, START_DATE="2012-01-01",
                      END_DATE="2012-12-31", FORECAST=10, N_ENSEMBLES=5,
                      WARMUP=365, WINDOW=30, PRE="scalar", POST=True,
                      device=None, GEFS_DIR=None, STATE_PATH=None,
                      skip_existing=False):
    """Generate/save a daily sequence of NPZ forecasts for one basin.

    Args: GAGE_ID scalar; model, data_loader, config; basin_pool (B,)
          ordered exactly like dataset's basin axis; model callbacks;
          START_DATE/END_DATE dates; FORECAST=L; N_ENSEMBLES=M; WARMUP int;
          PRE 'scalar'|'cdf'; POST bool; optional device/GEFS_DIR/STATE_PATH;
          skip_existing bool.
    Returns: dict with count of saved/skipped/failed initializations and paths.
    Writes (M,L) forecast per initialization day, not any CRPS metrics.
    """
    if device is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if GEFS_DIR is None:
        GEFS_DIR = ("/beegfs/sets/aw-ciroh/common/met_fcsts/"
                    "gefs_v12/camels_gII/camels671/zn_processed_hcst")
    if PRE not in ("scalar", "cdf"):
        raise ValueError("PRE must be 'scalar' or 'cdf': the existing GEFS restart "
                         "callback assumes an active bias-correction method")
    if STATE_PATH is None:
        STATE_PATH = os.path.join(
            config["sim_dir"], "simulation_gefs",
            "save_states_restart", "model_states.pt"
        )
    root = forecast_root(config, START_DATE, END_DATE, PRE, WINDOW, POST)
    folder = root / f"{int(GAGE_ID):08d}"
    folder.mkdir(parents=True, exist_ok=True)
    basin_idx = [int(b) for b in basin_pool].index(int(GAGE_ID))
    init_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    failed_dates = []
    n_saved = n_skipped = 0
    print(f"GEFS forecast for basin {int(GAGE_ID):08d}; "
          f"{len(init_dates)} initializations; output: {folder}")

    for pred_date in init_dates:
        path = forecast_file_path(root, GAGE_ID, pred_date)
        if skip_existing and path.exists():
            n_skipped += 1
            continue
        try:
            ensemble = _forecast_one_date(
                basin_id=int(GAGE_ID), basin_idx=basin_idx,
                pred_date=pred_date, forecast_days=FORECAST,
                n_ensembles=N_ENSEMBLES, warmup=WARMUP, window=WINDOW,
                pre=PRE, post=POST, model=model, data_loader=data_loader,
                config=config, device=device, gefs_dir=GEFS_DIR,
                state_path=STATE_PATH, startid_endid=startid_endid,
                run_segment=run_segment,
                run_warm_forecasts_restart=run_warm_forecasts_restart,
                post_processing=post_processing,
            )
            save_forecast_npz(root, GAGE_ID, pred_date, ensemble)
            n_saved += 1
        except Exception as exc:
            print(f"FAILED {pred_date.date()}: {exc}")
            failed_dates.append({"init_date": str(pred_date.date()),
                                 "error": str(exc)})

    failed_path = folder / "failed_dates.csv"
    pd.DataFrame(failed_dates, columns=["init_date", "error"]).to_csv(
        failed_path, index=False
    )
    print(f"Saved {n_saved}; skipped {n_skipped}; failed {len(failed_dates)}")
    return {
        "basin_id": f"{int(GAGE_ID):08d}",
        "save_dir": str(folder), "n_total": len(init_dates),
        "n_saved": n_saved, "n_skipped": n_skipped,
        "n_failed": len(failed_dates), "failed_dates": failed_dates,
        "failed_csv": str(failed_path),
    }


# ===========================================================================
# 7. Additional GEFS forecast verification metrics
# ===========================================================================



def _check_metric_inputs(ensemble_mean, target_flow):
    """Validate deterministic ensemble-mean predictions and observations.

        Args: ensemble_mean ndarray (I,L), target_flow ndarray (I,L).
        Returns: pair of float ndarrays (I,L).

    """
    mean = np.asarray(ensemble_mean, dtype=float)
    obs = np.asarray(target_flow, dtype=float)
    if mean.ndim != 2 or mean.shape != obs.shape or 0 in mean.shape:
        raise ValueError("Expected matching, nonempty (n_init,n_leads) arrays")
    return mean, obs

def _metric_valid(ensemble_mean, target_flow, lead, min_valid_cases):
    """Check enough finite deterministic ensemble-mean cases for one lead.

    Args: ensemble_mean (I,L), target_flow (I,L), lead int (0..L-1),
        min_valid_cases int.
    Returns: bool mask (I,); all False if not enough valid cases.
    """
    valid = (np.isfinite(ensemble_mean[:, lead])
             & np.isfinite(target_flow[:, lead]))
    return valid if valid.sum() >= min_valid_cases else np.zeros_like(valid)


def calculate_gefs_ns(ensemble_mean, target_flow, MIN_VALID_CASES=30):
    """Nash-Sutcliffe efficiency of ensemble mean for each lead.

    Args: ensemble_mean ndarray (I,L), target_flow ndarray (I,L);
          MIN_VALID_CASES int. Returns ndarray (L,).
    """
    ensemble_mean, target_flow = _check_metric_inputs(ensemble_mean, target_flow)
    out = np.full(ensemble_mean.shape[1], np.nan)
    for j in range(len(out)):
        valid = _metric_valid(ensemble_mean, target_flow, j, MIN_VALID_CASES)
        if not valid.any():
            continue
        pred, obs = ensemble_mean[valid, j], target_flow[valid, j]
        denom = np.sum((obs - obs.mean()) ** 2)
        if denom > 0:
            out[j] = 1 - np.sum((pred - obs) ** 2) / denom
    return out


def calculate_gefs_pbias(ensemble_mean, target_flow, MIN_VALID_CASES=30):
    """Percentage bias by lead (Alfieri sign: + means underprediction).

    Args: ensemble_mean ndarray (I,L), target_flow ndarray (I,L);
          MIN_VALID_CASES int. Returns ndarray (L,).
    """
    ensemble_mean, target_flow = _check_metric_inputs(ensemble_mean, target_flow)
    out = np.full(ensemble_mean.shape[1], np.nan)
    for j in range(len(out)):
        valid = _metric_valid(ensemble_mean, target_flow, j, MIN_VALID_CASES)
        if valid.any():
            pred, obs = ensemble_mean[valid, j], target_flow[valid, j]
            if not np.isclose(obs.mean(), 0, atol=1e-12):
                out[j] = 100 * (obs.mean() - pred.mean()) / obs.mean()
    return out


def calculate_gefs_cv(ensemble_mean, target_flow, MIN_VALID_CASES=30):
    """Coefficient of variation of RMSE for each lead: RMSE/mean(qobs).

    Args: ensemble_mean ndarray (I,L), target_flow ndarray (I,L);
          MIN_VALID_CASES int. Returns ndarray (L,).
    """
    ensemble_mean, target_flow = _check_metric_inputs(ensemble_mean, target_flow)
    out = np.full(ensemble_mean.shape[1], np.nan)
    for j in range(len(out)):
        valid = _metric_valid(ensemble_mean, target_flow, j, MIN_VALID_CASES)
        if valid.any():
            pred, obs = ensemble_mean[valid, j], target_flow[valid, j]
            if not np.isclose(obs.mean(), 0, atol=1e-12):
                out[j] = np.sqrt(np.mean((pred - obs) ** 2)) / obs.mean()
    return out


def plot_gefs_metric(values, metric_name, save_dir=None, START_DATE=None,
                     END_DATE=None, basin_id=None, percentiles=None,
                     show_plot=True):
    """Plot a single lead-wise metric with optional basin percentiles.

    Args: values ndarray (L,), metric_name str, percentiles optional
          tuple of four ndarrays (L,); save_dir Path/str|None.
    Returns: Path to PNG, or None if not saved.
    """
    import matplotlib.pyplot as plt
    values = np.asarray(values, dtype=float)
    leads = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(leads, values, color="black", linewidth=2.5, label=metric_name)
    if percentiles is not None:
        p05, p25, p75, p95 = [np.asarray(x, float) for x in percentiles]
        ax.fill_between(leads, p05, p95, color="lightgray", alpha=.75, label="5–95%")
        ax.fill_between(leads, p25, p75, color="gray", alpha=.65, label="25–75%")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlim(.5, len(leads) + .5)
    ax.set_xticks(leads)
    ax.set_xlabel("Lead time [days]")
    ax.set_ylabel(metric_name)
    title = f"GEFS Forecast {metric_name}"
    if basin_id is not None:
        title += f"\nBasin {int(basin_id):08d}"
    if START_DATE is not None and END_DATE is not None:
        title += f" ({START_DATE} to {END_DATE})"
    ax.set_title(title)
    if percentiles is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    saved_path = None
    if save_dir is not None:
        saved_path = Path(save_dir) / (
            "GEFS_" + metric_name.replace(" ", "_").replace("/", "_")
            .replace("%", "pct") + ".png"
        )
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return saved_path


def print_gefs_summary(target_basins, gefs_results, failed_basins):
    """Print forecast-generation completion statistics.

    Args: target_basins sequence of K IDs, gefs_results mapping basin->dict,
          failed_basins mapping/list. Returns: None.
    """
    from datetime import datetime
    print("\n========== FORECAST SUMMARY ==========")
    print("Requested basins:", len(target_basins))
    print("Completed basin calls:", len(gefs_results))
    print("Basins with failures:", len(failed_basins))
    for label, key in (("New forecasts", "n_saved"),
                       ("Skipped forecasts", "n_skipped"),
                       ("Failed forecast dates", "n_failed")):
        print(label + ":", sum(r[key] for r in gefs_results.values()))
    print("Failed basins:", failed_basins)
    print("End time:", datetime.now())
