#!/usr/bin/env python3
"""
Generate bar plots from evaluation stats CSV (stats_Context_*_allsamples.csv).

Produces:
  - Aggregate mean/median MAE (or MSE/MAPE) by model across datasets
  - Grouped bars: per-dataset, one bar per model (optional max_datasets)
  - Migas-1.5 win rate per dataset (migas15_win_pct)
  - Improvement over timeseries baseline per dataset
  - ELO ratings bar chart (if multielo available)
  - Single-dataset model comparison (all models for one chosen dataset or average)

Usage:
  python scripts/plot_bars.py --results_dir ./results/suite/context_64
  python scripts/plot_bars.py --results_dir ./results/suite/context_64 --metric mean_mae --out_dir ./results/suite/context_64/report
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from migaseval.baselines.registry import MODEL_DISPLAY_NAMES

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 9
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def discover_stats_csv(results_dir: Path) -> Path | None:
    """Find stats_Context_*_allsamples.csv in results_dir."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return None
    candidates = list(results_dir.glob("stats_Context_*_allsamples.csv"))
    if not candidates:
        return None
    return sorted(candidates)[0]


def infer_models_from_csv(df: pd.DataFrame, metric: str = "mean_mae") -> list[str]:
    """Infer model names from columns like migas15_mean_mae, chronos_univar_mean_mae."""
    suffix = f"_{metric}"
    models = []
    for c in df.columns:
        if c.endswith(suffix):
            name = c[: -len(suffix)]
            if name and name not in models:
                models.append(name)
    return models


def get_display_name(key: str, display_names: dict[str, str] | None) -> str:
    """Human-readable label for model or dataset."""
    if display_names and key in display_names:
        return display_names[key]
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    return key.replace("_", " ").strip()


def plot_aggregate_metric_by_model(
    df: pd.DataFrame,
    models: list[str],
    metric: str,
    out_dir: Path,
    model_display: dict[str, str] | None = None,
) -> None:
    """One bar per model: mean of metric across all datasets."""
    col = f"{metric}"
    if col not in df.columns:
        # try per-model column
        pass
    # Per-model columns like migas15_mean_mae
    values = []
    labels = []
    for m in models:
        c = f"{m}_{metric}"
        if c in df.columns:
            vals = df[c].dropna()
            if len(vals) > 0:
                values.append(vals.mean())
                labels.append(get_display_name(m, model_display))
            else:
                values.append(np.nan)
                labels.append(get_display_name(m, model_display))
        else:
            values.append(np.nan)
            labels.append(get_display_name(m, model_display))

    valid = [i for i, v in enumerate(values) if not np.isnan(v)]
    if not valid:
        return
    labels = [labels[i] for i in valid]
    values = [values[i] for i in valid]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    x = np.arange(len(labels))
    bars = ax.bar(
        x, values, color="#2ca02c", edgecolor="black", linewidth=0.8, width=0.7
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"Aggregate {metric.replace('_', ' ').title()} by Model (mean over datasets)"
    )
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / f"bar_aggregate_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_grouped_metric_by_dataset(
    df: pd.DataFrame,
    models: list[str],
    metric: str,
    out_dir: Path,
    max_datasets: int = 12,
    dataset_display: dict[str, str] | None = None,
    model_display: dict[str, str] | None = None,
) -> None:
    """Grouped bars: each group = one dataset, bars = models."""
    n_ds = len(df)
    if n_ds == 0:
        return
    # Optionally take top N datasets by row order or by avg metric
    plot_df = df.head(max_datasets)
    datasets = plot_df["dataset_name"].tolist()
    n_ds = len(datasets)
    n_models = len(models)
    x = np.arange(n_ds)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_ds * 0.8), 5))
    for i, m in enumerate(models):
        c = f"{m}_{metric}"
        if c not in plot_df.columns:
            continue
        vals = plot_df[c].values
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=get_display_name(m, model_display))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_display_name(d, dataset_display) for d in datasets],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Dataset (grouped by model)")
    ax.legend(loc="upper right", ncol=2 if n_models > 4 else 1)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / f"bar_grouped_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_migas15_win_rate_per_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    dataset_display: dict[str, str] | None = None,
) -> None:
    """Bar chart of migas15_win_pct per dataset."""
    if "migas15_win_pct" not in df.columns:
        return
    df = df.dropna(subset=["migas15_win_pct"])
    if len(df) == 0:
        return
    datasets = df["dataset_name"].tolist()
    values = df["migas15_win_pct"].values

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 0.35), 5))
    x = np.arange(len(datasets))
    colors = ["#2ca02c" if v >= 50 else "#d62728" for v in values]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_display_name(d, dataset_display) for d in datasets],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Migas-1.5 Win Rate (%)")
    ax.set_title("Migas-1.5 vs Timeseries-Only Win Rate per Dataset")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "bar_migas15_win_pct.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_improvement_per_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    dataset_display: dict[str, str] | None = None,
) -> None:
    """Bar chart of improvement_pct_mean per dataset (Migas-1.5 vs timeseries)."""
    if "improvement_pct_mean" not in df.columns:
        return
    df = df.dropna(subset=["improvement_pct_mean"])
    if len(df) == 0:
        return
    # Sort by improvement descending
    df = df.sort_values("improvement_pct_mean", ascending=False)
    datasets = df["dataset_name"].tolist()
    values = df["improvement_pct_mean"].values

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 0.35), 5))
    x = np.arange(len(datasets))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_display_name(d, dataset_display) for d in datasets],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Migas-1.5 vs Timeseries-Only: Mean MAE Improvement per Dataset")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "bar_improvement_pct.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_elo_bars(
    df: pd.DataFrame,
    models: list[str],
    metric: str,
    out_dir: Path,
    model_display: dict[str, str] | None = None,
) -> None:
    """ELO ratings bar chart (requires multielo)."""
    try:
        from multielo import MultiElo
    except ImportError:
        return
    # Build rankings per row: for each dataset row, rank models by metric (lower = better)
    metric_col = f"_{metric}"
    rankings = []
    for _, row in df.iterrows():
        row_rank = []
        for m in models:
            c = f"{m}_{metric}"
            if c in row and pd.notna(row[c]):
                row_rank.append((m, float(row[c])))
        if len(row_rank) >= 2:
            row_rank.sort(key=lambda x: x[1])
            rankings.append([m for m, _ in row_rank])

    if not rankings:
        return
    elo = MultiElo(k_value=32, d_value=400)
    base_rating = 1500
    ratings = {m: float(base_rating) for m in models}
    for rank in rankings:
        current = np.array([ratings[m] for m in rank])
        new = elo.get_new_ratings(current)
        for m, r in zip(rank, new):
            ratings[m] = r

    labels = [get_display_name(m, model_display) for m in models]
    values = [ratings[m] for m in models]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    x = np.arange(len(labels))
    colors = ["#2ca02c" if m == "migas15" else "#1f77b4" for m in models]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8, width=0.7)
    ax.axhline(y=base_rating, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ELO Rating")
    ax.set_title("Model ELO Ratings (from per-dataset rankings)")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / f"bar_elo_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_single_dataset_models(
    df: pd.DataFrame,
    models: list[str],
    metric: str,
    out_dir: Path,
    dataset_name: str | None = None,
    model_display: dict[str, str] | None = None,
) -> None:
    """One bar per model for a single dataset (or average across datasets)."""
    if dataset_name:
        row = df[df["dataset_name"] == dataset_name]
        if row.empty:
            return
        row = row.iloc[0]
        title_suffix = get_display_name(dataset_name, None)
    else:
        row = df.mean(numeric_only=True)
        title_suffix = "Average over datasets"

    values = []
    labels = []
    for m in models:
        c = f"{m}_{metric}"
        if c in row and pd.notna(row[c]):
            values.append(float(row[c]))
            labels.append(get_display_name(m, model_display))

    if not values:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#1f77b4", edgecolor="black", linewidth=0.8, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Model — {title_suffix}")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    suffix = dataset_name or "average"
    safe = re.sub(r"[^\w\-]", "_", suffix)[:40]
    fig.savefig(
        out_dir / f"bar_single_{safe}_{metric}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def run(
    results_dir: str | Path,
    out_dir: str | Path | None = None,
    metric: str = "mean_mae",
    max_datasets: int = 12,
    config: str | None = None,
    single_dataset: str | None = None,
) -> bool:
    """
    Generate bar plots from evaluation stats in results_dir.
    Call this from post_eval.py or other orchestration code.
    Returns True on success, False if matplotlib missing, no stats CSV, or no models.
    """
    if not HAS_MPL:
        print("matplotlib is required. Install with: pip install matplotlib")
        return False

    results_dir = Path(results_dir)
    csv_path = discover_stats_csv(results_dir)
    if csv_path is None:
        print(f"No stats_Context_*_allsamples.csv found in {results_dir}")
        return False

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Stats CSV is empty.")
        return False

    models = infer_models_from_csv(df, metric)
    if not models:
        print(f"No model columns found for metric {metric}")
        return False

    out_dir = Path(out_dir) if out_dir else results_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_display = None
    model_display = None
    if config and Path(config).exists():
        import yaml

        with open(config) as f:
            cfg = yaml.safe_load(f) or {}
        dataset_display = cfg.get("dataset_display_names")
        model_display = cfg.get("model_display_names")

    print(f"Using stats: {csv_path}")
    print(f"Models: {models}")
    print(f"Output: {out_dir}")

    plot_aggregate_metric_by_model(df, models, metric, out_dir, model_display)
    plot_grouped_metric_by_dataset(
        df,
        models,
        metric,
        out_dir,
        max_datasets=max_datasets,
        dataset_display=dataset_display,
        model_display=model_display,
    )
    plot_migas15_win_rate_per_dataset(df, out_dir, dataset_display)
    plot_improvement_per_dataset(df, out_dir, dataset_display)
    plot_elo_bars(df, models, metric, out_dir, model_display)
    # bar_single_average_* would duplicate bar_aggregate_* (same data), so only run for a specific dataset
    if single_dataset:
        plot_single_dataset_models(
            df,
            models,
            metric,
            out_dir,
            dataset_name=single_dataset,
            model_display=model_display,
        )

    print("Bar plots saved to", out_dir)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bar plots from evaluation stats CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing stats_Context_*_allsamples.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results_dir/report)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_mae",
        choices=[
            "mean_mae",
            "median_mae",
            "mean_mse",
            "median_mse",
            "mean_mape",
            "median_mape",
        ],
        help="Metric column suffix for model columns",
    )
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=12,
        help="Max datasets in grouped bar plot",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML with dataset_display_names and model_display_names",
    )
    parser.add_argument(
        "--single_dataset",
        type=str,
        default=None,
        help="If set, also plot single-dataset model comparison for this dataset",
    )
    args = parser.parse_args()

    ok = run(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        metric=args.metric,
        max_datasets=args.max_datasets,
        config=args.config,
        single_dataset=args.single_dataset,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
