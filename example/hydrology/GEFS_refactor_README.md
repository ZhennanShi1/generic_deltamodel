# Refactor: `tempfunctions.py` + `gefs_crpss.py`

## Files

- **`gefs_crpss.py`**: daily forecast NPZ paths, saving/loading, observation date alignment, CRPS and CRPSS equations, summary/CSV/plotting, NS/Pbias/CV, plus a daily forecasting wrapper.
- **`tempfunctions_refactored.py`**: original `tempfunctions.py` with the duplicated CRPSS, verification, and NPZ-writing workflow removed. GEFS forcing, bias correction, model restart, and calibration routines are retained.
- **`test_gefs_crpss.py`**: automated synthetic tests. They verify the exact ensemble CRPS, lead-date alignment, saving/loading, and equality of scores with your original `_gefs_basin_crpss()`.

The original attached file is not modified. In your project, back it up, then rename `tempfunctions_refactored.py` to `tempfunctions.py`. Put `gefs_crpss.py` in the same Python import directory.

## 1. Notebook imports

```python
import importlib
import tempfunctions
import gefs_crpss as gefs
importlib.reload(tempfunctions)
importlib.reload(gefs)
```

## 2. Build observed reference data

```python
basin_results = gefs.build_observed_basin_results(
    data_loader=data_loader,
    config=config,
    loader_basin_pool=loader_basin_pool,
    selected_basins=eval_basins,
    WARMUP=365,
)
```

## 3. Evaluate EXISTING forecast NPZs (no rerun of the model required)

```python
crpss_results = gefs.run_gefs_crpss(
    mode=MODE,
    config=config,
    basin_results=basin_results,
    basin_pool=eval_basins,
    GAGE_ID=eval_basins[0] if MODE == "single" else None,
    START_DATE=START_DATE,
    END_DATE=END_DATE,
    PRE=PRE,
    WINDOW=WINDOW,
    POST=POST,
    YEAR=2012,
    N_LEADS=10,
    N_ENSEMBLES=5,
    LEAD1_OFFSET=0,
    MIN_VALID_CASES=30,
    REFERENCE="observed",
    CLIMATOLOGY_YEARS=None,
    show_plot=True,
)
display(crpss_results["summary_df"])
```

You can retain the remainder of your existing notebook cell unchanged; change the module prefix `tempfunctions.` to `gefs.` on CRPSS-related calls.

## 4. Forecast generation and NPZ saving

The GEFS model/restart routines stay in `tempfunctions.py`. Their outputs are now saved using the NPZ functions in `gefs_crpss.py`. To call the refactored forecast-generation wrapper:

```python
result = gefs.run_gefs_one_year(
    GAGE_ID=GAGE_ID,
    model=model,
    data_loader=data_loader,
    config=config,
    basin_pool=loader_basin_pool,
    startid_endid=tempfunctions.startid_endid,
    run_segment=tempfunctions.run_segment,
    run_warm_forecasts_restart=tempfunctions.run_warm_forecasts_restart,
    post_processing=tempfunctions.post_processing,
    START_DATE=START_DATE,
    END_DATE=END_DATE,
    FORECAST=10,
    N_ENSEMBLES=5,
    WARMUP=365,
    WINDOW=WINDOW,
    PRE=PRE,
    POST=POST,
    skip_existing=True,
)
```

Use the same `basin_pool`, `GEFS_DIR`, `STATE_PATH`, and `device` arguments as your original call if they differ from the defaults. **`basin_pool` above must be the full basin order in `data_loader`, not the 36-basin evaluation subset.**

To manually save a single already-generated forecast:

```python
root = gefs.forecast_root(config, START_DATE, END_DATE, PRE, WINDOW, POST)
gefs.save_forecast_npz(root, GAGE_ID, PRED_DATE, ens_preds)
```

This retains the original path and keys:

```text
<config['sim_dir']>/GEFS_1year_forecasts/
    2012-01-01_2012-12-31_pre-scalar_w30_post-True/
        01487000/forecast_20120101.npz

NPZ keys: forecast (n_members, n_leads), init_date, basin_id
```

## 5. Read saved forecasts

```python
fc, init_dates = gefs.load_basin_forecasts(
    root, GAGE_ID, year=2012, n_members=5, n_leads=10
)
print(fc.shape)  # (n_init_dates, 10, 5)
```

## 6. Other GEFS metrics

```python
ensemble_mean, target_flow = gefs.load_gefs_metric_data(
    GAGE_ID=GAGE_ID,
    config=config,
    basin_results=basin_results,
    START_DATE=START_DATE,
    END_DATE=END_DATE,
    PRE=PRE, WINDOW=WINDOW, POST=POST,
    N_LEADS=10, N_ENSEMBLES=5, LEAD1_OFFSET=0,
    REFERENCE="observed",
)
ns = gefs.calculate_gefs_ns(ensemble_mean, target_flow)
pbias = gefs.calculate_gefs_pbias(ensemble_mean, target_flow)
cv = gefs.calculate_gefs_cv(ensemble_mean, target_flow)
```

## Conventions and behavior

- **Day 1 = initialization date** when `LEAD1_OFFSET=0`; verification dates are `init + LT - 1`.
- Persistence reference = reference discharge on **the day before** initialization for this convention.
- CRPS is calculated across **all five ensemble members per forecast case**, then averaged over valid initialization dates **separately for each LT**.
- `CRPS_ref_ad` remains the mean absolute deviation around the climatological mean of the chosen historical sample. With `CLIMATOLOGY_YEARS=None` it uses *all* available dates, even those in the evaluation year. Change `CLIMATOLOGY_YEARS` if you need strictly pre-evaluation climatology.
- Shared `valid` mask ensures forecast and persistence scores use identical cases. A lead is scored for CRPSS only if `n_valid >= MIN_VALID_CASES` and reference CRPS is positive.
- The refactor **preserves the original reference definitions**; it does not fix anomalies from the underlying GEFS forecast simulations.

## Run synthetic tests

```bash
python -m pytest -q test_gefs_crpss.py
```
