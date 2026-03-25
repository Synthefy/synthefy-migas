"""Visualize target and extracted covariate columns from hydro reservoir feature CSVs.

Generates per-file plots showing:
  1. Target (y_t) with covariate overlay (planned_mw, unplanned_mw)
  2. Covariate sparsity: fraction of non-zero values in rolling windows
  3. Cross-correlation between target and each covariate at various lags
  4. Covariate value distributions (histograms, zero vs non-zero)

Usage:
    uv run python scripts/visualize_covariates.py
    uv run python scripts/visualize_covariates.py --files data/NO_1_daily_hydro_reservoir_features.csv
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from migaseval.plotting_utils import apply_migas_style

apply_migas_style()

COV_COLS = ["planned_count", "planned_mw", "unplanned_count", "unplanned_mw"]
MW_COLS = ["planned_mw", "unplanned_mw"]
COUNT_COLS = ["planned_count", "unplanned_count"]
TARGET = "y_t"


def plot_file(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path, parse_dates=["t"])
    fname = Path(csv_path).stem
    for col in COV_COLS:
        df[col] = df[col].fillna(0.0)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    dates = df["t"]
    target = df[TARGET].values

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(5, 1, hspace=0.35)
    ax_ts0 = fig.add_subplot(gs[0])
    ax_ts1 = fig.add_subplot(gs[1], sharex=ax_ts0)
    ax_ts2 = fig.add_subplot(gs[2], sharex=ax_ts0)
    ax_xcorr = fig.add_subplot(gs[3])
    ax_hist = fig.add_subplot(gs[4])
    axes = [ax_ts0, ax_ts1, ax_ts2, ax_xcorr, ax_hist]
    fig.suptitle(fname.replace("_features", ""), fontsize=16, y=0.98)

    # --- Panel 1: Target + MW covariates ---
    ax = axes[0]
    ax.plot(dates, target, color="#2c3e50", linewidth=0.7, label=TARGET)
    ax.set_ylabel("y_t", color="#2c3e50")
    ax.tick_params(axis="y", labelcolor="#2c3e50")

    ax2 = ax.twinx()
    ax2.bar(dates, df["planned_mw"].values, width=1.5, alpha=0.5, color="#e67e22", label="planned_mw")
    ax2.bar(dates, df["unplanned_mw"].values, width=1.5, alpha=0.5, color="#e74c3c", label="unplanned_mw", bottom=df["planned_mw"].values)
    ax2.set_ylabel("MW", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.set_title("Target vs outage MW")

    # --- Panel 2: Covariate sparsity (rolling 30-day non-zero fraction) ---
    ax = axes[1]
    window = 30
    for col, color in zip(COV_COLS, ["#e67e22", "#f39c12", "#e74c3c", "#c0392b"]):
        nonzero_frac = (df[col] > 0).rolling(window, min_periods=1).mean()
        ax.plot(dates, nonzero_frac, label=col, color=color, linewidth=1)
    ax.set_ylabel("Fraction non-zero")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Covariate sparsity (rolling {window}-day non-zero fraction)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # --- Panel 3: Target + count covariates ---
    ax = axes[2]
    ax.plot(dates, target, color="#2c3e50", linewidth=0.7, label=TARGET)
    ax.set_ylabel("y_t", color="#2c3e50")
    ax.tick_params(axis="y", labelcolor="#2c3e50")

    ax2 = ax.twinx()
    ax2.step(dates, df["planned_count"].values, where="mid", color="#3498db", alpha=0.7, linewidth=0.8, label="planned_count")
    ax2.step(dates, df["unplanned_count"].values, where="mid", color="#e74c3c", alpha=0.7, linewidth=0.8, label="unplanned_count")
    ax2.set_ylabel("Count", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.set_title("Target vs outage counts")

    # --- Panel 4: Cross-correlation at different lags ---
    ax = axes[3]
    max_lag = 30
    lags = np.arange(-max_lag, max_lag + 1)
    for col, color in zip(MW_COLS, ["#e67e22", "#e74c3c"]):
        cov_vals = df[col].values
        t_centered = target - target.mean()
        c_centered = cov_vals - cov_vals.mean()
        norm = np.sqrt(np.sum(t_centered ** 2) * np.sum(c_centered ** 2))
        if norm < 1e-12:
            continue
        xcorr = np.array([
            np.sum(t_centered[max(0, lag): len(target) + min(0, lag)] *
                   c_centered[max(0, -lag): len(target) - max(0, lag)]) / norm
            for lag in lags
        ])
        ax.plot(lags, xcorr, label=col, color=color, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (days, positive = covariate leads target)")
    ax.set_ylabel("Cross-correlation")
    ax.legend(fontsize=8)
    ax.set_title("Cross-correlation: target vs MW covariates")

    # --- Panel 5: Value distributions ---
    ax = axes[4]
    for i, (col, color) in enumerate(zip(MW_COLS, ["#e67e22", "#e74c3c"])):
        vals = df[col].values
        n_zero = np.sum(vals == 0)
        n_nonzero = np.sum(vals > 0)
        pct_zero = n_zero / len(vals) * 100

        nonzero_vals = vals[vals > 0]
        if len(nonzero_vals) > 0:
            ax.hist(nonzero_vals, bins=40, alpha=0.5, color=color,
                    label=f"{col} (non-zero, {pct_zero:.0f}% are zero)")
        else:
            ax.text(0.5, 0.5 - i * 0.1, f"{col}: 100% zero", transform=ax.transAxes, fontsize=10, color=color)

    ax.set_xlabel("MW value (excluding zeros)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.set_title("Distribution of non-zero covariate values")

    for ax_ts in [ax_ts0, ax_ts1, ax_ts2]:
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha="right")
    out_path = Path(out_dir) / f"{fname}_covariates.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # --- Print summary stats ---
    print(f"\n  {fname} — covariate summary:")
    for col in COV_COLS:
        vals = df[col].values
        n_nonzero = np.sum(vals > 0)
        pct_nonzero = n_nonzero / len(vals) * 100
        print(f"    {col:>20s}: {pct_nonzero:5.1f}% non-zero  |  "
              f"mean={vals.mean():.1f}  max={vals.max():.0f}  "
              f"mean(when>0)={vals[vals > 0].mean():.1f}" if n_nonzero > 0
              else f"    {col:>20s}: 100% zero")

    corr_target = df[COV_COLS].corrwith(df[TARGET])
    print(f"\n    Pearson correlation with {TARGET}:")
    for col in COV_COLS:
        print(f"      {col:>20s}: {corr_target[col]:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize covariate columns")
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    if args.files is None:
        args.files = sorted(glob(str(data_dir / "*_daily_hydro_reservoir_features.csv")))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for fpath in args.files:
        plot_file(fpath, args.output_dir)


if __name__ == "__main__":
    main()
