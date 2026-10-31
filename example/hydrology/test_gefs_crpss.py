"""Small synthetic regression tests for gefs_crpss.py (run: pytest -q)."""
import numpy as np
import pandas as pd
import pytest

import gefs_crpss as g


def test_crps_exact_example():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    assert g.crps_single(x, 3) == pytest.approx(0.4)
    fc = x.reshape(1, 1, 5)
    assert g.ensemble_crps_cases(fc, np.array([[3.]]))[0, 0] == pytest.approx(0.4)


def test_date_alignment_day1_and_persistence():
    dates = pd.date_range("2011-12-30", "2012-01-15")
    series = pd.Series(np.arange(len(dates), dtype=float), index=dates)
    targets, initial = g.align_reference(series, [pd.Timestamp("2012-01-02")], 3)
    assert targets.tolist() == [[3., 4., 5.]]
    assert initial.tolist() == [2.]
    # Alternative convention: Day 1 = first day after initialization
    targets2, initial2 = g.align_reference(series, [pd.Timestamp("2012-01-02")], 3, 1)
    assert targets2.tolist() == [[4., 5., 6.]]
    assert initial2.tolist() == [3.]


def test_small_npz_pipeline(tmp_path):
    config = {"sim_dir": str(tmp_path)}
    root = g.forecast_root(config)
    dt = pd.date_range("2012-01-01", periods=3)
    fc = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]])
    for d in dt:
        g.save_forecast_npz(root, "01487000", d, fc)
    assert g.forecast_file_path(root, "01487000", dt[0]).exists()
    loaded, inits = g.load_basin_forecasts(root, "01487000", year=2012,
                                          n_members=5, n_leads=2)
    assert loaded.shape == (3, 2, 5)
    assert inits.tolist() == dt.tolist()

    qdates = pd.date_range("2011-12-31", "2012-01-09")
    qref = pd.Series(np.linspace(1, 5, len(qdates)), index=qdates)
    basin_results = {"01487000": {"dates": qdates, "obs": qref.to_numpy()}}
    result = g.evaluate_gefs_basin("01487000", basin_results, root, year=2012,
                                    n_leads=2, min_valid_cases=2)
    assert (result["n_valid"] == 3).all()
    assert np.isfinite(result["CRPS_forecast"]).all()
    assert np.isfinite(result["CRPS_ref_pf"]).all()
    assert result["CRPS_ref_ad"][0] == result["CRPS_ref_ad"][1]

    out = g.run_gefs_crpss(
        "single", config, basin_results, GAGE_ID="01487000", N_LEADS=2,
        MIN_VALID_CASES=2, show_plot=False,
    )
    assert out["results_df"].shape[0] == 2
    assert out["summary_df"].shape[0] == 2
    from pathlib import Path
    for key in ("figure_path", "csv_path", "summary_csv"):
        assert Path(out[key]).exists()


def test_missing_reference_and_threshold():
    fc = np.array([[[1., 2.], [2., 3.]],
                   [[2., 3.], [3., 4.]]])
    dates = pd.to_datetime(["2012-01-02", "2012-01-03"])
    qdates = pd.date_range("2012-01-01", "2012-01-05")
    qref = pd.Series(np.ones(len(qdates)), index=qdates)
    result = g.calculate_basin_crpss(fc, dates, qref, min_valid_cases=30)
    assert result["n_valid"].tolist() == [2, 2]
    assert np.isnan(result["CRPSS_pf"]).all()
    assert np.isfinite(result["CRPS_forecast"]).all()


def test_existing_equation_equivalence():
    rng = np.random.default_rng(11)
    fc = rng.uniform(0, 10, size=(11, 7, 5))
    obs = rng.uniform(0, 10, size=(11, 7))
    fast = g.ensemble_crps_cases(fc, obs)
    loop = np.array([[g.crps_single(fc[i, j], obs[i, j]) for j in range(7)]
                     for i in range(11)])
    np.testing.assert_allclose(fast, loop)


def test_manual_reference_equations_match_pipeline(tmp_path):
    """Regression of Eq. (7), Eq. (8), and exact CRPS over multiple cases."""
    root = tmp_path / 'forecasts'
    init_dates = pd.date_range('2012-01-01', periods=4)
    rng = np.random.default_rng(99)
    fc_array = rng.uniform(1, 3, size=(4, 10, 5))
    for d, fc in zip(init_dates, fc_array):
        g.save_forecast_npz(root, '01487000', d, fc.T)

    qdates = pd.date_range('2011-12-31', '2012-02-01')
    qref = rng.uniform(.3, 4, size=len(qdates))
    observed = pd.Series(qref, index=qdates)
    ref = {'01487000': {'dates': qdates, 'obs': qref}}
    result = g.evaluate_gefs_basin('01487000', ref, root, year=2012,
                                    n_leads=10, n_ensembles=5,
                                    lead1_offset=0, min_valid_cases=2)
    for j in range(10):
        target = np.array([observed[d + pd.Timedelta(days=j)] for d in init_dates])
        initial = np.array([observed[d - pd.Timedelta(days=1)] for d in init_dates])
        forecast_crps = np.mean([
            np.mean(np.abs(fc_array[i, j] - target[i]))
            - 0.5 * np.mean(np.abs(fc_array[i, j, :, None]
                                   - fc_array[i, j, None, :]))
            for i in range(4)
        ])
        persistence_crps = np.mean(np.abs(target - initial))
        clim_crps = np.mean(np.abs(qref - qref.mean()))
        assert result['CRPS_forecast'][j] == pytest.approx(forecast_crps)
        assert result['CRPS_ref_pf'][j] == pytest.approx(persistence_crps)
        assert result['CRPS_ref_ad'][j] == pytest.approx(clim_crps)
        assert result['CRPSS_pf'][j] == pytest.approx(1 - forecast_crps / persistence_crps)
        assert result['CRPSS_ad'][j] == pytest.approx(1 - forecast_crps / clim_crps)
