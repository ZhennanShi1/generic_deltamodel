#!/usr/bin/env python
"""
Generate multi-page PDF of basin/station diagnostic plots.

Works for both:
  - Streamflow (CAMELS basins) — use --names-file with camels_name.txt
  - SWE (SNOTEL stations) — use --names-file with snotel_topo.csv

For each basin/station, produces a page with:
  - Header: name, ID, metrics (KGE, NSE, r, alpha, bias)
  - Panel 1: Monthly mean time series (obs vs predicted)
  - Panel 2: Daily time series (obs vs predicted)
  - Panel 3: Mean annual cycle (mean by calendar month)

Usage:
    python plot_basin_diagnostics.py <run_dir> [options]

Examples:
    # Streamflow
    python plot_basin_diagnostics.py runs/streamflow_prism_0704_104320 \
        --n-basins 50 --sort-by none \
        --names-file /beegfs/sets/aw-ciroh/common/datasets/camels/CAMELS_US/camels_attributes_v2.0/camels_name.txt

    # SWE
    python plot_basin_diagnostics.py runs/swe_prism_0704_120000 \
        --sort-by none \
        --names-file snotel_topo.csv
"""

import sys
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy import stats


def find_results_pickle(run_dir):
    """Find the test results pickle file, searching nested directories."""
    run_dir = Path(run_dir)
    candidates = []
    candidates.extend(run_dir.glob("test/*.p"))
    candidates.extend(run_dir.glob("test/model_epoch*/*.p"))

    if not candidates:
        raise FileNotFoundError(f"No pickle files found in {run_dir}/test/")

    # Prefer the largest file
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def find_metrics_csv(run_dir):
    """Find test_metrics.csv as fallback for metrics."""
    run_dir = Path(run_dir)
    candidates = []
    candidates.extend(run_dir.glob("test/test_metrics.csv"))
    candidates.extend(run_dir.glob("test/model_epoch*/test_metrics.csv"))
    if candidates:
        return candidates[0]
    return None


def load_csv_metrics(csv_path):
    """Load metrics from test_metrics.csv into a dict of {basin_id: {metric: value}}."""
    df = pd.read_csv(csv_path, dtype={'basin': str})
    metrics_dict = {}
    for _, row in df.iterrows():
        bid = str(row['basin'])
        m = {}
        for col in df.columns:
            if col == 'basin':
                continue
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                m[col] = float(val)
        metrics_dict[bid] = m
    return metrics_dict


def load_all_basins(pkl_path, run_dir):
    """
    Load NH test results pickle and extract obs/sim time series + metrics.
    Falls back to test_metrics.csv if metrics not in pickle.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Try loading CSV metrics as fallback
    csv_metrics = {}
    csv_path = find_metrics_csv(run_dir)
    if csv_path is not None:
        csv_metrics = load_csv_metrics(csv_path)

    basins = {}
    for basin_id, basin_dict in data.items():
        bid = str(basin_id)

        # Navigate into frequency key ('1D')
        if '1D' in basin_dict:
            inner = basin_dict['1D']
        else:
            first_key = list(basin_dict.keys())[0]
            inner = basin_dict[first_key]

        if 'xr' not in inner:
            continue

        ds = inner['xr']

        # Extract obs and sim variable names
        var_names = list(ds.data_vars)
        obs_vars = [v for v in var_names if '_obs' in v]
        sim_vars = [v for v in var_names if '_sim' in v]
        if not obs_vars or not sim_vars:
            continue

        obs_var = obs_vars[0]
        sim_var = sim_vars[0]

        # Get arrays — squeeze time_step dimension if present
        obs = ds[obs_var].values.squeeze()
        sim = ds[sim_var].values.squeeze()
        dates = pd.DatetimeIndex(ds['date'].values)

        # Collect metrics — first try pickle, then CSV fallback
        metrics = {}
        for key, val in inner.items():
            if key == 'xr':
                continue
            if isinstance(val, (int, float, np.floating)):
                metrics[key] = float(val)

        # If pickle had no metrics, use CSV
        if not metrics and bid in csv_metrics:
            metrics = csv_metrics[bid]

        # Compute r and bias if not already present
        mask = np.isfinite(obs) & np.isfinite(sim)
        if mask.sum() > 10:
            o, s = obs[mask], sim[mask]
            if 'r' not in metrics and 'Pearson-r' not in metrics:
                r, _ = stats.pearsonr(o, s)
                metrics['r'] = r
            elif 'Pearson-r' in metrics and 'r' not in metrics:
                metrics['r'] = metrics['Pearson-r']
            if 'Bias' not in metrics:
                metrics['Bias'] = 100 * (np.sum(s - o) / np.sum(o)) if np.sum(o) != 0 else np.nan
            if 'alpha' not in metrics and 'Alpha-NSE' not in metrics:
                metrics['alpha'] = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan

        basins[bid] = {
            'dates': dates,
            'obs': obs,
            'sim': sim,
            'metrics': metrics,
        }

    return basins


def load_names(names_file):
    """Load basin/station names from CAMELS or SNOTEL format."""
    names_path = Path(names_file)
    with open(names_path, 'r') as f:
        header = f.readline()

    if ';' in header:
        # CAMELS format: gauge_id;huc_02;gauge_name
        names_df = pd.read_csv(names_path, sep=';', dtype={0: str})
        id_col = names_df.columns[0]
        name_col = [c for c in names_df.columns if 'name' in c.lower()][0]
        names_df[id_col] = names_df[id_col].astype(str).str.strip()
    else:
        # SNOTEL/CSV format: Station ID,State,Network,Station Name,...
        names_df = pd.read_csv(names_path, dtype={0: str})
        names_df.columns = names_df.columns.str.strip()
        id_candidates = ['Station ID', 'station_id', 'stationid', 'id', 'gauge_id']
        name_candidates = ['Station Name', 'station_name', 'gauge_name', 'name']
        id_col = next((c for c in names_df.columns if c in id_candidates), names_df.columns[0])
        name_col = next((c for c in names_df.columns if c in name_candidates), names_df.columns[3])
        names_df[id_col] = names_df[id_col].astype(str).str.strip()

    return dict(zip(names_df[id_col], names_df[name_col].astype(str).str.strip()))


def plot_basin_page(dates, obs, sim, basin_id, metrics, target_label, basin_name=None):
    """Create a single basin/station diagnostic page with 3 panels."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5),
                             gridspec_kw={'height_ratios': [1, 1, 0.8]})
    fig.subplots_adjust(hspace=0.35, top=0.88, bottom=0.08, left=0.10, right=0.95)

    df = pd.DataFrame({'obs': obs, 'sim': sim}, index=dates)
    df = df.dropna(subset=['obs'])

    obs_color = '#2ca02c'
    sim_color = '#1f77b4'

    # --- Header ---
    nse = metrics.get('NSE', np.nan)
    kge = metrics.get('KGE', np.nan)
    r = metrics.get('r', metrics.get('Pearson-r', np.nan))
    alpha = metrics.get('Alpha-NSE', metrics.get('alpha', np.nan))
    bias = metrics.get('Bias', np.nan)

    if basin_name:
        title_line = f"{basin_name}  (ID={basin_id})"
    else:
        title_line = f"ID: {basin_id}"

    header = (
        f"{title_line}\n"
        f"KGE={kge:.3f}   r={r:.3f}   \u03b1={alpha:.3f}   "
        f"NSE={nse:.3f}   Bias={bias:+.1f}%"
    )
    fig.suptitle(header, fontsize=11, fontweight='bold', y=0.96,
                 fontfamily='monospace')

    # --- Panel 1: Monthly mean ---
    ax1 = axes[0]
    if len(df) > 0:
        monthly = df.resample('ME').mean()
        ax1.plot(monthly.index, monthly['obs'], color=obs_color, linewidth=1.0,
                 label='Obs', alpha=0.9)
        ax1.plot(monthly.index, monthly['sim'], color=sim_color, linewidth=1.0,
                 label='Sim', alpha=0.9)
    ax1.set_ylabel(target_label, fontsize=9)
    ax1.set_title('Monthly', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=8)

    # --- Panel 2: Daily ---
    ax2 = axes[1]
    if len(df) > 0:
        ax2.plot(df.index, df['obs'], color=obs_color, linewidth=0.4,
                 label='Obs', alpha=0.7)
        ax2.plot(df.index, df['sim'], color=sim_color, linewidth=0.4,
                 label='Sim', alpha=0.7)
    ax2.set_ylabel(target_label, fontsize=9)
    ax2.set_title('Daily', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right', ncol=2)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=8)

    # --- Panel 3: Mean Annual Cycle ---
    ax3 = axes[2]
    df_valid = df.dropna()
    if len(df_valid) > 0:
        clim_obs = df_valid['obs'].groupby(df_valid.index.month).mean()
        clim_sim = df_valid['sim'].groupby(df_valid.index.month).mean()

        months = range(1, 13)
        month_labels = ['J', 'F', 'M', 'A', 'M', 'J',
                        'J', 'A', 'S', 'O', 'N', 'D']

        ax3.plot(list(months), [clim_obs.get(m, np.nan) for m in months],
                 'o-', color=obs_color, linewidth=1.5, markersize=4, label='Obs')
        ax3.plot(list(months), [clim_sim.get(m, np.nan) for m in months],
                 'o-', color=sim_color, linewidth=1.5, markersize=4, label='Sim')

        ax3.set_xticks(list(months))
        ax3.set_xticklabels(month_labels)
        ax3.set_xlabel('Month', fontsize=9)

    ax3.set_ylabel(target_label, fontsize=9)
    ax3.set_title('Mean Annual Cycle', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right', ncol=2)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelsize=8)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate basin/station diagnostic PDF from NeuralHydrology results')
    parser.add_argument('run_dir', help='Path to NH run directory')
    parser.add_argument('--n-basins', type=int, default=None,
                        help='Number of basins/stations to plot (default: all)')
    parser.add_argument('--sort-by', type=str, default='KGE',
                        help='Sort by this metric (best first), or "none" for original order')
    parser.add_argument('--target-label', type=str, default=None,
                        help='Y-axis label (default: auto-detect from run name)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF path (default: <run_dir>/basin_diagnostics.pdf)')
    parser.add_argument('--names-file', type=str, default=None,
                        help='Path to names file — camels_name.txt (semicolon-sep) '
                             'or snotel_topo.csv (comma-sep)')

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    # Auto-detect target label from run directory name
    if args.target_label:
        target_label = args.target_label
    elif 'swe' in run_dir.name.lower():
        target_label = 'SWE (mm)'
    else:
        target_label = 'Flow (mm/day)'

    # Auto-detect task type for logging
    task_type = 'station' if 'swe' in run_dir.name.lower() else 'basin'

    # Load names if provided
    basin_names = {}
    if args.names_file:
        basin_names = load_names(args.names_file)
        print(f"  Loaded {len(basin_names)} {task_type} names")

    # Find and load results
    print(f"Loading results from: {run_dir}")
    pkl_path = find_results_pickle(run_dir)
    print(f"  Pickle: {pkl_path}")

    basins = load_all_basins(pkl_path, run_dir)
    print(f"  Loaded {len(basins)} {task_type}s with obs/sim data")

    if not basins:
        print(f"No valid {task_type}s found.")
        sys.exit(1)

    # Check how many have metrics
    n_with_metrics = sum(1 for b in basins.values() if b['metrics'])
    if n_with_metrics < len(basins):
        print(f"  WARNING: {len(basins) - n_with_metrics} {task_type}s have no metrics")

    # Sort
    sort_key = args.sort_by
    if sort_key.lower() == 'none':
        sorted_ids = list(basins.keys())
    else:
        sorted_ids = sorted(
            basins.keys(),
            key=lambda b: basins[b]['metrics'].get(sort_key, -999),
            reverse=True
        )

    if args.n_basins:
        sorted_ids = sorted_ids[:args.n_basins]

    # Output path
    out_path = Path(args.output) if args.output else run_dir / f"{task_type}_diagnostics.pdf"

    print(f"Generating PDF: {len(sorted_ids)} {task_type}s, sorted by {sort_key}")
    print(f"Output: {out_path}")

    with PdfPages(str(out_path)) as pdf:
        for i, bid in enumerate(sorted_ids):
            bd = basins[bid]

            if (i + 1) % 25 == 0 or i == 0:
                sort_val = bd['metrics'].get(sort_key, np.nan)
                sort_str = f"{sort_val:.3f}" if np.isfinite(sort_val) else "N/A"
                print(f"  [{i+1}/{len(sorted_ids)}] {bid} ({sort_key}={sort_str})")

            fig = plot_basin_page(
                bd['dates'], bd['obs'], bd['sim'],
                bid, bd['metrics'],
                target_label=target_label,
                basin_name=basin_names.get(bid, None)
            )
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

    print(f"\nDone! {out_path}")


if __name__ == '__main__':
    main()