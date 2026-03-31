"""TabPFN-TS evaluation on pre-windowed CSVs (ttfm_review format).

Each CSV represents one evaluation window (context + prediction horizon).
The directory layout is:

    <data_root>/
        context_<N>/
            <series_name>/
                summary_<idx>.csv   # N + pred_len rows

The first N rows are the context; the last pred_len rows are the horizon to forecast.

Usage:
    # Run all modes, all contexts, all series found under data_root
    uv run python scripts/tabpfn_ts_ttfm_eval.py \\
        --data-root /data/ttfm_review/hydropower_features_subset \\
        --output-dir results/tabpfn_ttfm

    # Specific modes
    uv run python scripts/tabpfn_ts_ttfm_eval.py \\
        --data-root /data/ttfm_review/hydropower_features_subset \\
        --output-dir results/tabpfn_ttfm \\
        --modes no_leak planned_leak

    # Specific contexts and series
    uv run python scripts/tabpfn_ts_ttfm_eval.py \\
        --data-root /data/ttfm_review/hydropower_features_subset \\
        --output-dir results/tabpfn_ttfm \\
        --context-lens 128 256 \\
        --series NO_1_daily_hydro_reservoir SE_3_daily_hydro_reservoir

Output:
    <output_dir>/context_<N>/<series_name>/<mode>.npz

    Each .npz contains:
        preds        (n_windows, pred_len)  predictions  [normalized]
        gts          (n_windows, pred_len)  ground truth [normalized]
        preds_raw    (n_windows, pred_len)  predictions  [original scale]
        gts_raw      (n_windows, pred_len)  ground truth [original scale]
        last_ctx     (n_windows,)           last context value [normalized]
        last_ctx_raw (n_windows,)           last context value [original scale]
        window_ids   (n_windows,)           summary index from filename
        target_mu    (n_windows,)           per-window z-score mean
        target_sigma (n_windows,)           per-window z-score std
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="tabpfn_time_series")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TABPFN_2_5_CHECKPOINT = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
TABPFN_WEIGHTS_DIR = "/tmp/tabpfn"
TABPFN_CONFIG = {"model_path": f"{TABPFN_WEIGHTS_DIR}/{TABPFN_2_5_CHECKPOINT}"}

PRED_LEN = 16

ALL_COV_COLS = ["planned_count", "planned_mw", "unplanned_count", "unplanned_mw"]
PLANNED_COV_COLS = ["planned_count", "planned_mw"]
TARGET_COL = "y_t"

ALL_MODES = {
    "no_leak": [],
    "planned_leak": PLANNED_COV_COLS,
    "all_leak": ALL_COV_COLS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zscore_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def build_tsdfs(
    ctx_vals: np.ndarray,
    ctx_covs: np.ndarray,
    fut_covs: np.ndarray | None,
    ctx_dates: pd.DatetimeIndex,
    fut_dates: pd.DatetimeIndex,
    leak_cols: list[str],
):
    """Build train/test TimeSeriesDataFrames with per-window z-score normalisation.

    Returns (train_tsdf, test_tsdf, target_mu, target_sigma).
    """
    from tabpfn_time_series import TimeSeriesDataFrame

    pred_len = len(fut_dates)

    target_mu = float(ctx_vals.mean())
    target_sigma = float(ctx_vals.std())
    if target_sigma < 1e-8:
        target_sigma = 1.0

    train_dict: dict = {
        "item_id": "series_0",
        "timestamp": ctx_dates,
        "target": (ctx_vals - target_mu) / target_sigma,
    }
    test_dict: dict = {
        "item_id": "series_0",
        "timestamp": fut_dates,
        "target": np.full(pred_len, np.nan),
    }

    if leak_cols:
        cov_mu, cov_sigma = _zscore_stats(ctx_covs)
        for i, col in enumerate(ALL_COV_COLS):
            if col in leak_cols:
                train_dict[col] = (ctx_covs[:, i] - cov_mu[i]) / cov_sigma[i]
                if fut_covs is not None:
                    test_dict[col] = (fut_covs[:, i] - cov_mu[i]) / cov_sigma[i]
                else:
                    test_dict[col] = np.zeros(pred_len)

    train_tsdf = TimeSeriesDataFrame.from_data_frame(pd.DataFrame(train_dict))
    test_tsdf = TimeSeriesDataFrame.from_data_frame(pd.DataFrame(test_dict))

    return train_tsdf, test_tsdf, target_mu, target_sigma


def load_predictor():
    from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

    print("Loading TabPFN-TS 2.5 (LOCAL mode) ...")
    predictor = TabPFNTimeSeriesPredictor(
        tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_config=TABPFN_CONFIG,
        tabpfn_output_selection="mean",
    )
    return predictor


def window_id_from_path(csv_path: Path) -> int:
    """Extract the numeric index from summary_<idx>.csv."""
    m = re.search(r"(\d+)", csv_path.stem)
    return int(m.group(1)) if m else -1


def load_window_csv(csv_path: Path, context_len: int, pred_len: int):
    """Load one window CSV; return (ctx_vals, ctx_covs, gt_vals, fut_covs, ctx_dates, fut_dates).

    The CSV must have at least context_len + pred_len rows.
    Missing covariate columns are filled with 0.
    """
    df = pd.read_csv(csv_path, parse_dates=["t"])

    for col in ALL_COV_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Drop leading NaN target rows
    valid_mask = ~df[TARGET_COL].isna()
    df = df[valid_mask].reset_index(drop=True)

    if len(df) < context_len + pred_len:
        return None  # not enough rows

    ctx_vals = df[TARGET_COL].values[:context_len].astype(np.float32)
    ctx_covs = df[ALL_COV_COLS].values[:context_len].astype(np.float32)
    gt_vals = df[TARGET_COL].values[context_len: context_len + pred_len].astype(np.float32)
    fut_covs = df[ALL_COV_COLS].values[context_len: context_len + pred_len].astype(np.float32)
    ctx_dates = pd.DatetimeIndex(df["t"].values[:context_len])
    fut_dates = pd.DatetimeIndex(df["t"].values[context_len: context_len + pred_len])

    return ctx_vals, ctx_covs, gt_vals, fut_covs, ctx_dates, fut_dates


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_series(
    series_dir: Path,
    context_len: int,
    predictor,
    modes: dict[str, list[str]],
    output_dir: Path,
):
    """Evaluate all windows for one (context_len, series) pair across requested modes.

    Saves one .npz per mode to output_dir/<mode>.npz.
    """
    from tabpfn_time_series import FeatureTransformer
    from tabpfn_time_series.features import RunningIndexFeature

    csv_files = sorted(series_dir.glob("summary_*.csv"), key=window_id_from_path)
    if not csv_files:
        print(f"  [SKIP] no CSVs in {series_dir}")
        return

    # Accumulators per mode
    acc: dict[str, dict[str, list]] = {
        m: {
            "preds": [], "gts": [], "preds_raw": [], "gts_raw": [],
            "last_ctx": [], "last_ctx_raw": [],
            "window_ids": [], "target_mu": [], "target_sigma": [],
        }
        for m in modes
    }

    n_total = len(csv_files)
    for w_idx, csv_path in enumerate(csv_files):
        window_data = load_window_csv(csv_path, context_len, PRED_LEN)
        if window_data is None:
            print(f"  [WARN] {csv_path.name}: insufficient rows, skipping")
            continue

        ctx_vals, ctx_covs, gt_vals, fut_covs, ctx_dates, fut_dates = window_data
        win_id = window_id_from_path(csv_path)

        for mode_name, leak_cols in modes.items():
            fut = fut_covs if leak_cols else None

            train_tsdf, test_tsdf, t_mu, t_sigma = build_tsdfs(
                ctx_vals, ctx_covs, fut, ctx_dates, fut_dates, leak_cols
            )
            ft = FeatureTransformer([RunningIndexFeature()])
            train_ft, test_ft = ft.transform(train_tsdf, test_tsdf)

            result = predictor.predict(train_ft, test_ft)
            forecast = result["target"].values.astype(np.float32)  # (pred_len,)

            gt_norm = (gt_vals - t_mu) / t_sigma
            last_norm = (float(ctx_vals[-1]) - t_mu) / t_sigma

            a = acc[mode_name]
            a["preds"].append(forecast)
            a["gts"].append(gt_norm)
            a["preds_raw"].append(forecast * t_sigma + t_mu)
            a["gts_raw"].append(gt_vals)
            a["last_ctx"].append(last_norm)
            a["last_ctx_raw"].append(float(ctx_vals[-1]))
            a["window_ids"].append(win_id)
            a["target_mu"].append(t_mu)
            a["target_sigma"].append(t_sigma)

        if (w_idx + 1) % 20 == 0 or w_idx == n_total - 1:
            print(f"    Window {w_idx + 1}/{n_total}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for mode_name, a in acc.items():
        if not a["preds"]:
            continue
        out_path = output_dir / f"{mode_name}.npz"
        np.savez(
            out_path,
            preds=np.stack(a["preds"]),
            gts=np.stack(a["gts"]),
            preds_raw=np.stack(a["preds_raw"]),
            gts_raw=np.stack(a["gts_raw"]),
            last_ctx=np.array(a["last_ctx"]),
            last_ctx_raw=np.array(a["last_ctx_raw"]),
            window_ids=np.array(a["window_ids"]),
            target_mu=np.array(a["target_mu"]),
            target_sigma=np.array(a["target_sigma"]),
        )
        print(f"    Saved {out_path}  ({len(a['preds'])} windows)")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_contexts(data_root: Path, context_lens: list[int] | None) -> list[tuple[int, Path]]:
    """Return [(context_len, context_dir), ...] under data_root."""
    found = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"context_(\d+)", d.name)
        if not m:
            continue
        ctx = int(m.group(1))
        if context_lens is None or ctx in context_lens:
            found.append((ctx, d))
    return found


def discover_series(context_dir: Path, series_filter: list[str] | None) -> list[Path]:
    """Return series subdirectories, optionally filtered by name."""
    found = []
    for d in sorted(context_dir.iterdir()):
        if not d.is_dir():
            continue
        if series_filter is None or d.name in series_filter:
            found.append(d)
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TabPFN-TS evaluation on pre-windowed ttfm_review CSVs"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Root directory containing context_<N> subdirectories",
    )
    parser.add_argument(
        "--output-dir", default="results/tabpfn_ttfm",
        help="Directory to write .npz result files (default: results/tabpfn_ttfm)",
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=list(ALL_MODES.keys()) + ["all"],
        default=["all"],
        help=(
            "Leak modes to run. Options: no_leak planned_leak all_leak all "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--context-lens", type=int, nargs="+", default=None,
        help="Which context lengths to process (default: all found under data_root)",
    )
    parser.add_argument(
        "--series", nargs="+", default=None,
        help="Which series names to process (default: all found in each context dir)",
    )
    args = parser.parse_args()

    # Resolve modes
    if "all" in args.modes:
        selected_modes = ALL_MODES
    else:
        selected_modes = {m: ALL_MODES[m] for m in args.modes}

    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)

    if not data_root.exists():
        print(f"ERROR: data_root does not exist: {data_root}")
        return

    contexts = discover_contexts(data_root, args.context_lens)
    if not contexts:
        print(f"No context_<N> directories found under {data_root}")
        return

    print(f"Data root   : {data_root}")
    print(f"Output dir  : {output_root}")
    print(f"Modes       : {list(selected_modes.keys())}")
    print(f"Contexts    : {[c for c, _ in contexts]}")
    print(f"Series filter: {args.series or 'all'}")

    predictor = load_predictor()

    for ctx_len, ctx_dir in contexts:
        series_dirs = discover_series(ctx_dir, args.series)
        if not series_dirs:
            print(f"\n[context_{ctx_len}] No matching series directories found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  context_len={ctx_len}  |  {len(series_dirs)} series")
        print(f"{'='*60}")

        for series_dir in series_dirs:
            series_name = series_dir.name
            print(f"\n  >> {series_name}")

            out_dir = output_root / f"context_{ctx_len}" / series_name
            evaluate_series(
                series_dir=series_dir,
                context_len=ctx_len,
                predictor=predictor,
                modes=selected_modes,
                output_dir=out_dir,
            )

    print(f"\nDone. Results written to {output_root}")


if __name__ == "__main__":
    main()
