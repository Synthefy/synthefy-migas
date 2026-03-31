"""TabPFN-TS univariate rolling backtest with future-leaked covariates.

Leakage modes:
    no_leak        — no covariates (pure univariate)
    planned_leak   — planned_count + planned_mw as tabular features
    all_leak       — all 4 covariates as tabular features

Usage:
    uv run python scripts/tabpfn_ts_backtest.py
    uv run python scripts/tabpfn_ts_backtest.py \
        --context-lens 64 128 --stride 16 \
        --files data/NO_1_daily_hydro_reservoir_features.csv
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="tabpfn_time_series")

TABPFN_2_5_CHECKPOINT = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
TABPFN_WEIGHTS_DIR = "/tmp/tabpfn"
TABPFN_CONFIG = {"model_path": f"{TABPFN_WEIGHTS_DIR}/{TABPFN_2_5_CHECKPOINT}"}

ALL_COV_COLS = ["planned_count", "planned_mw", "unplanned_count", "unplanned_mw"]
PLANNED_COV_COLS = ["planned_count", "planned_mw"]
TARGET_COL = "y_t"

LEAK_MODES = {
    "no_leak": [],
    "planned_leak": PLANNED_COV_COLS,
    "all_leak": ALL_COV_COLS,
}


def _zscore_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def build_tsdfs(
    ctx_vals: np.ndarray,
    ctx_covs: np.ndarray,
    fut_covs: np.ndarray | None,
    pred_len: int,
    leak_cols: list[str],
):
    """Build train/test TimeSeriesDataFrames with z-score normalization.

    Returns (train_tsdf, test_tsdf, target_mu, target_sigma).
    """
    from tabpfn_time_series import TimeSeriesDataFrame

    seq_len = len(ctx_vals)
    ctx_end = pd.Timestamp("2024-01-01") + pd.Timedelta(days=seq_len - 1)
    ctx_dates = pd.date_range(end=ctx_end, periods=seq_len, freq="D")
    fut_dates = pd.date_range(
        start=ctx_end + pd.Timedelta(days=1), periods=pred_len, freq="D",
    )

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


def compute_metrics(preds: np.ndarray, gts: np.ndarray, last_ctx: np.ndarray) -> dict:
    errors = preds - gts
    mae_per = np.mean(np.abs(errors), axis=1)
    mse_per = np.mean(errors ** 2, axis=1)

    denom = np.abs(gts) + 1e-8
    mape_per = np.mean(np.abs(errors) / denom, axis=1) * 100

    pred_dir = np.sign(preds[:, 0] - last_ctx)
    gt_dir = np.sign(gts[:, 0] - last_ctx)
    valid = gt_dir != 0
    dir_acc = float((pred_dir[valid] == gt_dir[valid]).mean() * 100) if valid.any() else float("nan")

    mean_mse = float(np.mean(mse_per))
    return {
        "MAE (mean)": float(np.mean(mae_per)),
        "MAE (median)": float(np.median(mae_per)),
        "MSE (mean)": mean_mse,
        "MSE (median)": float(np.median(mse_per)),
        "RMSE (mean)": float(np.sqrt(mean_mse)),
        "MAPE % (mean)": float(np.mean(mape_per)),
        "MAPE % (median)": float(np.median(mape_per)),
        "Dir. acc % (step 1)": dir_acc,
    }


def load_predictor():
    from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

    print("Loading TabPFN-TS 2.5 (LOCAL mode) ...")
    predictor = TabPFNTimeSeriesPredictor(
        tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_config=TABPFN_CONFIG,
        tabpfn_output_selection="mean",
    )
    return predictor


def run_backtest(
    csv_path: str,
    predictor,
    context_len: int,
    pred_len: int,
    stride: int,
) -> dict:
    """Run all leakage modes for one CSV at one context length."""
    from tabpfn_time_series import FeatureTransformer
    from tabpfn_time_series.features import RunningIndexFeature

    df = pd.read_csv(csv_path)
    fname = Path(csv_path).stem

    for col in ALL_COV_COLS:
        df[col] = df[col].fillna(0.0)

    target = df[TARGET_COL].values.astype(np.float32)
    covariates = df[ALL_COV_COLS].values.astype(np.float32)

    valid_mask = ~np.isnan(target)
    first_valid = int(np.argmax(valid_mask))
    target = target[first_valid:]
    covariates = covariates[first_valid:]

    n_total = len(target)
    n_windows = (n_total - context_len - pred_len) // stride + 1
    if n_windows <= 0:
        print(f"  {fname}: not enough data for context_len={context_len}")
        return {}

    print(f"\n{'='*60}")
    print(f"  {fname}  |  ctx={context_len}  |  stride={stride}  |  {n_windows} windows")
    print(f"{'='*60}")

    ft = FeatureTransformer([RunningIndexFeature()])

    results = {}
    for mode_name, leak_cols in LEAK_MODES.items():
        preds_list = []
        gts_list = []
        last_ctx_list = []

        for i in range(n_windows):
            start = i * stride
            end = start + context_len

            ctx_vals = target[start:end]
            ctx_covs = covariates[start:end]
            gt = target[end: end + pred_len]

            if len(gt) < pred_len:
                break

            fut_covs = covariates[end: end + pred_len] if leak_cols else None

            train_tsdf, test_tsdf, t_mu, t_sigma = build_tsdfs(
                ctx_vals, ctx_covs, fut_covs, pred_len, leak_cols,
            )

            train_ft, test_ft = ft.transform(train_tsdf, test_tsdf)

            result = predictor.predict(train_ft, test_ft)
            forecast = result["target"].values.astype(np.float32)

            gt_norm = (gt - t_mu) / t_sigma

            preds_list.append(forecast)
            gts_list.append(gt_norm)
            last_ctx_list.append((float(ctx_vals[-1]) - t_mu) / t_sigma)

            if (i + 1) % 20 == 0 or i == n_windows - 1:
                print(f"  [{mode_name}] Window {i + 1}/{n_windows}")

        preds_arr = np.stack(preds_list)
        gts_arr = np.stack(gts_list)
        last_ctx_arr = np.array(last_ctx_list)

        metrics = compute_metrics(preds_arr, gts_arr, last_ctx_arr)
        results[mode_name] = metrics

        print(f"  {mode_name}: MAE={metrics['MAE (mean)']:.4f}  MSE={metrics['MSE (mean)']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TabPFN-TS univariate rolling backtest")
    parser.add_argument(
        "--context-lens", type=int, nargs="+",
        default=[32, 64, 128, 256, 384, 512],
    )
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--pred-len", type=int, default=16)
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    if args.files is None:
        args.files = sorted(glob(str(data_dir / "*_daily_hydro_reservoir_features.csv")))

    if not args.files:
        print("No feature CSV files found. Run extract_outage_features.py first.")
        return

    print(f"Context lengths: {args.context_lens}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Stride: {args.stride}")
    print(f"Leak modes: {list(LEAK_MODES.keys())}")
    print(f"Files: {[Path(f).name for f in args.files]}")

    predictor = load_predictor()

    all_results: dict = {}

    for fpath in args.files:
        fname = Path(fpath).stem
        all_results[fname] = {}
        for ctx_len in args.context_lens:
            file_results = run_backtest(
                fpath, predictor,
                context_len=ctx_len,
                pred_len=args.pred_len,
                stride=args.stride,
            )
            if file_results:
                all_results[fname][str(ctx_len)] = file_results

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"tabpfn_ts_stride{args.stride}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY  (MAE mean, normalized)")
    print("=" * 80)
    mode_names = list(LEAK_MODES.keys())
    for fname, ctx_dict in all_results.items():
        print(f"\n  {fname}")
        header = f"    {'ctx':>6s}" + "".join(f"  {m:>14s}" for m in mode_names)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for ctx_len in sorted(ctx_dict.keys(), key=int):
            modes = ctx_dict[ctx_len]
            vals = "".join(
                f"  {modes[m]['MAE (mean)']:>14.4f}" for m in mode_names
            )
            print(f"    {ctx_len:>6s}{vals}")


if __name__ == "__main__":
    main()
