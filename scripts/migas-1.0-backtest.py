"""Migas-1.0 rolling backtest via Synthefy forecast API.

Follows the same rolling-window protocol as the Chronos2/Toto/TabPFN scripts:
feeds raw values, renormalizes predictions to z-score space for metrics.

Leakage modes:
    univariate     — pure univariate (no covariates sent)
    no_leak        — covariates as metadata, leak_target=False
    planned_leak   — planned covariates leak_target=True
    all_leak       — all covariates leak_target=True

Requires SYNTHEFY_API_KEY in environment or .env file.

Usage:
    uv run python scripts/migas-1.0-backtest.py
    uv run python scripts/migas-1.0-backtest.py \
        --context-lens 64 128 --stride 16 \
        --files data/NO_1_daily_hydro_reservoir_features.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

MIGAS_API_URL = "https://forecast.synthefy.com/v2/forecast"
API_SUB_BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0

ALL_COV_COLS = ["planned_count", "planned_mw", "unplanned_count", "unplanned_mw"]
PLANNED_COV_COLS = ["planned_count", "planned_mw"]
TARGET_COL = "y_t"

LEAK_MODES = {
    "univariate": None,
    "no_leak": [],
    "planned_leak": PLANNED_COV_COLS,
    "all_leak": ALL_COV_COLS,
}


def _call_migas_api(rows: list[list[dict]], headers: dict, timeout: int = 120) -> list[dict]:
    """Send rows to Migas API with retries.

    Each row is [target_sample, cov1, cov2, ...].
    Returns target forecast dict for each row.
    """
    payload = {
        "model": "Migas-latest",
        "samples": rows,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                MIGAS_API_URL, json=payload, headers=headers, timeout=timeout,
            )
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                print(f"  Rate-limited (429), retrying in {wait:.0f}s ...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"\nMigas API error {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
            data = resp.json()
            return [row[0] for row in data["forecasts"]]
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  Timeout, retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Migas API: max retries exceeded")


def _build_row(
    window_idx: int,
    ctx_dates: list[str],
    fut_dates: list[str],
    ctx_vals: np.ndarray,
    ctx_covs: np.ndarray,
    fut_covs: np.ndarray,
    pred_len: int,
    leak_cols: list[str] | None,
) -> list[dict]:
    """Build one API row (target + optional covariate samples) for a window."""
    target_sample = {
        "sample_id": f"w{window_idx}_target",
        "history_timestamps": ctx_dates,
        "history_values": ctx_vals.tolist(),
        "target_timestamps": fut_dates,
        "target_values": [None] * pred_len,
        "forecast": True,
        "metadata": False,
        "leak_target": False,
        "column_name": TARGET_COL,
    }

    if leak_cols is None:
        return [target_sample]

    row = [target_sample]
    for i, col in enumerate(ALL_COV_COLS):
        should_leak = col in (leak_cols or [])
        cov_sample = {
            "sample_id": f"w{window_idx}_{col}",
            "history_timestamps": ctx_dates,
            "history_values": ctx_covs[:, i].tolist(),
            "target_timestamps": fut_dates,
            "target_values": fut_covs[:, i].tolist() if should_leak else [None] * pred_len,
            "forecast": False,
            "metadata": True,
            "leak_target": should_leak,
            "column_name": col,
        }
        row.append(cov_sample)

    return row


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


def run_backtest(
    csv_path: str,
    headers: dict,
    context_len: int,
    pred_len: int,
    stride: int,
) -> dict:
    """Run all leakage modes for one CSV at one context length."""
    df = pd.read_csv(csv_path)
    fname = Path(csv_path).stem

    for col in ALL_COV_COLS:
        df[col] = df[col].fillna(0.0)

    dates = df["t"].values
    target = df[TARGET_COL].values.astype(np.float64)
    covariates = df[ALL_COV_COLS].values.astype(np.float64)

    valid_mask = ~np.isnan(target)
    first_valid = int(np.argmax(valid_mask))
    dates = dates[first_valid:]
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

    results = {}
    for mode_name, leak_cols in LEAK_MODES.items():
        all_rows: list[list[dict]] = []
        window_stats: list[tuple[float, float, np.ndarray, float]] = []

        for wi in range(n_windows):
            start = wi * stride
            end = start + context_len

            ctx_vals = target[start:end]
            gt = target[end:end + pred_len]
            if len(gt) < pred_len:
                break

            ctx_covs = covariates[start:end]
            fut_covs = covariates[end:end + pred_len]

            mu = float(ctx_vals.mean())
            sigma = float(ctx_vals.std())
            if sigma < 1e-8:
                sigma = 1.0

            ctx_dates = [str(d) for d in dates[start:end]]
            fut_dates = [str(d) for d in dates[end:end + pred_len]]

            row = _build_row(
                wi, ctx_dates, fut_dates,
                ctx_vals, ctx_covs, fut_covs,
                pred_len, leak_cols,
            )
            all_rows.append(row)
            window_stats.append((mu, sigma, gt, float(ctx_vals[-1])))

        preds_list = []
        gts_list = []
        last_ctx_list = []

        for sb_start in range(0, len(all_rows), API_SUB_BATCH_SIZE):
            sb_end = min(sb_start + API_SUB_BATCH_SIZE, len(all_rows))
            sub_rows = all_rows[sb_start:sb_end]

            try:
                sub_results = _call_migas_api(sub_rows, headers)
            except (requests.HTTPError, RuntimeError) as exc:
                print(f"  [{mode_name}] API error at windows {sb_start}-{sb_end}: {exc}")
                sub_results = []
                for row in sub_rows:
                    last_val = row[0]["history_values"][-1] if row[0]["history_values"] else 0.0
                    sub_results.append({"values": [last_val] * pred_len})

            for j_local, result in enumerate(sub_results):
                j = sb_start + j_local
                mu, sigma, gt, last_val = window_stats[j]

                forecast_vals = result["values"]
                forecast_raw = np.array(
                    [v if v is not None else 0.0 for v in forecast_vals[:pred_len]],
                    dtype=np.float64,
                )

                preds_list.append(((forecast_raw - mu) / sigma).astype(np.float32))
                gts_list.append(((gt - mu) / sigma).astype(np.float32))
                last_ctx_list.append((last_val - mu) / sigma)

            done = sb_end
            if done % 20 == 0 or done == len(all_rows):
                print(f"  [{mode_name}] Window {done}/{len(all_rows)}")

        preds_arr = np.stack(preds_list)
        gts_arr = np.stack(gts_list)
        last_ctx_arr = np.array(last_ctx_list, dtype=np.float32)

        metrics = compute_metrics(preds_arr, gts_arr, last_ctx_arr)
        results[mode_name] = metrics

        print(f"  {mode_name}: MAE={metrics['MAE (mean)']:.4f}  MSE={metrics['MSE (mean)']:.4f}")

    return results


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Migas-1.0 rolling backtest via Synthefy API")
    parser.add_argument(
        "--context-lens", type=int, nargs="+",
        default=[32, 64, 128, 256, 384, 512],
    )
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--pred-len", type=int, default=16)
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    api_key = os.environ.get("SYNTHEFY_API_KEY")
    if not api_key:
        print("ERROR: SYNTHEFY_API_KEY not set in environment or .env")
        print("Set it via: export SYNTHEFY_API_KEY=your-key")
        print("Or add SYNTHEFY_API_KEY=... to your .env file")
        return

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    data_dir = Path(__file__).resolve().parent.parent / "data"
    if args.files is None:
        args.files = sorted(glob(str(data_dir / "*_daily_hydro_reservoir_features.csv")))

    if not args.files:
        print("No feature CSV files found. Run extract_outage_features.py first.")
        return

    print(f"API: {MIGAS_API_URL}")
    print(f"Context lengths: {args.context_lens}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Stride: {args.stride}")
    print(f"Leak modes: {list(LEAK_MODES.keys())}")
    print(f"Files: {[Path(f).name for f in args.files]}")

    all_results: dict = {}

    for fpath in args.files:
        fname = Path(fpath).stem
        all_results[fname] = {}
        for ctx_len in args.context_lens:
            file_results = run_backtest(
                fpath, headers,
                context_len=ctx_len,
                pred_len=args.pred_len,
                stride=args.stride,
            )
            if file_results:
                all_results[fname][str(ctx_len)] = file_results

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"migas_stride{args.stride}.json")
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
