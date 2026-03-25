"""TabPFN-TS univariate rolling backtest with future-leaked covariates.

Matches the evaluation approach in eval_utils.py: feeds raw-scale values to
TabPFN, then renormalizes predictions back to z-score space for metrics.

Leakage modes:
    univariate     — no covariates (pure univariate, same as no_leak for TabPFN)
    no_leak        — no covariates (TabPFN can't use past-only covariates)
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
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ALL_COV_COLS = ["planned_count", "planned_mw", "unplanned_count", "unplanned_mw"]
PLANNED_COV_COLS = ["planned_count", "planned_mw"]
TARGET_COL = "y_t"

LEAK_MODES = {
    "univariate": None,
    "no_leak": [],
    "planned_leak": PLANNED_COV_COLS,
    "all_leak": ALL_COV_COLS,
}


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

    print("Loading TabPFN-TS (LOCAL mode) ...")
    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
    return predictor


def run_backtest(
    csv_path: str,
    predictor,
    context_len: int,
    pred_len: int,
    stride: int,
    batch_size: int = 32,
) -> dict:
    """Run all leakage modes for one CSV at one context length."""
    from tabpfn_time_series import TimeSeriesDataFrame, FeatureTransformer
    from tabpfn_time_series.features import (
        RunningIndexFeature, CalendarFeature, AutoSeasonalFeature,
    )

    df = pd.read_csv(csv_path)
    fname = Path(csv_path).stem

    for col in ALL_COV_COLS:
        df[col] = df[col].fillna(0.0)

    target = df[TARGET_COL].values.astype(np.float64)
    covariates = df[ALL_COV_COLS].values.astype(np.float64)

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

    base_features = [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]
    context_end = pd.Timestamp.today().normalize()
    context_range = pd.date_range(end=context_end, periods=context_len, freq="D")
    fut_range = pd.date_range(
        start=context_end + pd.Timedelta(days=1), periods=pred_len, freq="D",
    )

    results = {}
    for mode_name, leak_cols in LEAK_MODES.items():
        preds_list = []
        gts_list = []
        last_ctx_list = []

        num_batches = (n_windows + batch_size - 1) // batch_size

        for bi in range(num_batches):
            w_start = bi * batch_size
            w_end = min((bi + 1) * batch_size, n_windows)
            bs = w_end - w_start

            train_records = []
            test_records = []
            mus = np.empty(bs, dtype=np.float64)
            sigmas = np.empty(bs, dtype=np.float64)
            batch_gts = []
            batch_last_ctx = []

            for j in range(bs):
                wi = w_start + j
                start = wi * stride
                end = start + context_len

                ctx_vals = target[start:end]
                gt = target[end: end + pred_len]

                mu = float(ctx_vals.mean())
                sigma = float(ctx_vals.std())
                if sigma < 1e-8:
                    sigma = 1.0
                mus[j] = mu
                sigmas[j] = sigma

                gt_norm = (gt - mu) / sigma
                batch_gts.append(gt_norm)
                batch_last_ctx.append((float(ctx_vals[-1]) - mu) / sigma)

                item_id = f"w{wi}"

                for t_idx, ts in enumerate(context_range):
                    rec = {"item_id": item_id, "timestamp": ts,
                           "target": float(ctx_vals[t_idx])}
                    if leak_cols:
                        ctx_covs = covariates[start:end]
                        for col in leak_cols:
                            ci = ALL_COV_COLS.index(col)
                            rec[col] = float(ctx_covs[t_idx, ci])
                    train_records.append(rec)

                for t_idx, ts in enumerate(fut_range):
                    rec = {"item_id": item_id, "timestamp": ts,
                           "target": np.nan}
                    if leak_cols:
                        fut_covs = covariates[end: end + pred_len]
                        for col in leak_cols:
                            ci = ALL_COV_COLS.index(col)
                            rec[col] = float(fut_covs[t_idx, ci])
                    test_records.append(rec)

            train_df = pd.DataFrame(train_records).set_index(["item_id", "timestamp"])
            test_df = pd.DataFrame(test_records).set_index(["item_id", "timestamp"])
            train_tsdf = TimeSeriesDataFrame(train_df)
            test_tsdf = TimeSeriesDataFrame(test_df)

            ft = FeatureTransformer(base_features)
            train_t, test_t = ft.transform(train_tsdf, test_tsdf)

            pred_df = predictor.predict(train_t, test_t)

            for j in range(bs):
                wi = w_start + j
                item_id = f"w{wi}"
                raw_pred = pred_df.loc[item_id]["target"].values[:pred_len].astype(np.float64)

                pred_norm = (raw_pred - mus[j]) / sigmas[j]

                preds_list.append(pred_norm.astype(np.float32))
                gts_list.append(batch_gts[j].astype(np.float32))
                last_ctx_list.append(batch_last_ctx[j])

            done = w_end
            if done % 20 == 0 or done == n_windows:
                print(f"  [{mode_name}] Window {done}/{n_windows}")

        preds_arr = np.stack(preds_list)
        gts_arr = np.stack(gts_list)
        last_ctx_arr = np.array(last_ctx_list, dtype=np.float32)

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
    parser.add_argument("--batch-size", type=int, default=32)
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
    print(f"Batch size: {args.batch_size}")
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
                batch_size=args.batch_size,
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
