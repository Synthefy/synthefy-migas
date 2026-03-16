#!/usr/bin/env python3
"""
Unified evaluation: Migas-1.5 vs baselines.

Baselines: Chronos (via Migas-1.5), TimesFM, Toto, TabPFN, Prophet, SARIMA.

Two modes:
    # From pre-cached summaries (no LLM needed)
    python -m migaseval.evaluation --summaries_dir ./results/test/context_64

    # From raw CSVs (generates summaries, needs LLM server)
    python -m migaseval.evaluation --datasets_dir ./data/test

    # Context length sweeping with baselines
    python -m migaseval.evaluation --summaries_dir ./results/test/context_64 \
        --context_lengths 32 64 128 256 384 --eval_timesfm --eval_prophet
"""

import csv as csv_mod
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm
from tabulate import tabulate

from migaseval.eval_utils import (
    evaluate_migas_precomputed,
    evaluate_timesfm_precomputed,
    evaluate_toto_precomputed,
    evaluate_tabpfn_precomputed,
    evaluate_prophet_precomputed,
    evaluate_sarima_precomputed,
    _crop_and_rescale,
    _load_preds,
    _save_preds,
    _has_preds,
    load_summaries,
    compute_metrics,
    generate_and_cache_summaries,
)
from migaseval.model import build_model
from migaseval.pipeline import _resolve_checkpoint_path

# ── Defaults ─────────────────────────────────────────────────────────────────
PRED_LEN = 16
BATCH_SIZE = 128
DEVICE = "cuda"
TEXT_EMBEDDER = "finbert"


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified Migas-1.5 evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summaries_dir",
        default=None,
        help="Directory with pre-cached summary JSONs. "
        "Each dataset has a subdirectory with summary_0.json, etc.",
    )
    parser.add_argument(
        "--datasets_dir",
        default=None,
        help="Directory with CSV/Parquet files. "
        "Summaries will be generated (needs LLM server).",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Output directory for prediction caches and result CSVs.",
    )
    parser.add_argument("--seq_len", type=int, default=384)
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--text_embedder", default=TEXT_EMBEDDER)
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths to evaluate at (e.g. 32 128 384). "
        "If not set, uses the full history length from summaries.",
    )
    parser.add_argument(
        "--eval_timesfm",
        action="store_true",
        help="Also evaluate TimesFM 2.5 baseline",
    )
    parser.add_argument(
        "--eval_toto",
        action="store_true",
        help="Also evaluate Toto baseline (univariate)",
    )
    parser.add_argument(
        "--eval_tabpfn",
        action="store_true",
        help="Also evaluate TabPFN 2.5 time-series baseline",
    )
    parser.add_argument(
        "--eval_prophet",
        action="store_true",
        help="Also evaluate Prophet baseline",
    )
    parser.add_argument(
        "--eval_sarima",
        action="store_true",
        help="Also evaluate Seasonal ARIMA (auto_arima) baseline",
    )
    parser.add_argument("--llm_base_url", default="http://localhost:8004/v1")
    parser.add_argument("--llm_model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    # ── Resolve summaries directory ───────────────────────────────────────
    if args.summaries_dir is None and args.datasets_dir is None:
        parser.error("At least one of --summaries_dir or --datasets_dir is required.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.datasets_dir is not None:
        # Generate summaries from CSVs, store in output_dir
        summaries_dir = os.path.join(args.output_dir, "summaries")
        print(f"Generating summaries from {args.datasets_dir} ...")
        generate_and_cache_summaries(
            datasets_dir=args.datasets_dir,
            summaries_dir=summaries_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
        )
        print(f"Summaries cached in {summaries_dir}\n")
    else:
        summaries_dir = args.summaries_dir

    # ── Build & load Migas-1.5 model ─────────────────────────────────────
    checkpoint_path = _resolve_checkpoint_path(
        "Synthefy/migas-1.5",
        filename="model.pt",
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    state_dict = ckpt.get("state_dict", ckpt)

    model = build_model(
        pred_len=args.pred_len,
        device=args.device,
        chronos_device=args.device,
        text_embedder=args.text_embedder,
        text_embedder_device=args.device,
        use_convex_combination=True,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys")
    model.eval()
    model.to(args.device)
    print("Model loaded.\n")

    # ── Discover datasets ────────────────────────────────────────────────
    summary_dirs = sorted(
        d
        for d in os.listdir(summaries_dir)
        if os.path.isdir(os.path.join(summaries_dir, d))
    )
    print(f"Found {len(summary_dirs)} datasets in {summaries_dir}\n")

    # ── Pre-load all dataset summaries once ──────────────────────────────
    all_datasets = {}
    for ds_name in summary_dirs:
        cached = load_summaries(summaries_dir, ds_name)
        if cached is None:
            print(f"  Skipped {ds_name} (no summaries)")
            continue
        all_datasets[ds_name] = cached  # (summaries, historic, forecast, means, stds)

    print(f"Loaded {len(all_datasets)} datasets with summaries\n")

    if not all_datasets:
        print("No datasets found. Check --summaries_dir or --datasets_dir.")
        return

    first_hist = list(all_datasets.values())[0][1]
    full_ctx = len(first_hist[0])
    if args.context_lengths is None:
        args.context_lengths = [full_ctx]

    # ── Loop over context lengths ────────────────────────────────────────
    for ctx_len in args.context_lengths:
        ctx_dir = os.path.join(args.output_dir, f"context_{ctx_len}")
        os.makedirs(ctx_dir, exist_ok=True)

        print("\n" + "#" * 80)
        print(f"  EVALUATING AT CONTEXT LENGTH = {ctx_len}")
        print("#" * 80 + "\n")

        test_set_name = os.path.basename(os.path.normpath(summaries_dir))
        csv_path = os.path.join(
            ctx_dir, f"results_{test_set_name}_ctx{ctx_len}.csv"
        )

        rows = []
        migas_wins, chronos_wins, ties = 0, 0, 0

        for ds_name in tqdm(all_datasets, desc=f"Eval ctx={ctx_len}"):
            summaries, historic, forecast, means, stds = all_datasets[ds_name]

            if ctx_len >= full_ctx:
                hist_eval = historic
                fcast_eval = forecast
                ctx_means = means
                ctx_stds = stds
            else:
                hist_eval, fcast_eval, ctx_means, ctx_stds = _crop_and_rescale(
                    historic, forecast, means, stds, ctx_len
                )

            hist_arr = np.array(hist_eval)
            gt_arr = np.array(fcast_eval)
            means_arr = np.array(ctx_means, dtype=np.float64)
            stds_arr = np.array(ctx_stds, dtype=np.float64)

            # ── helper to run-or-load a single model ─────────────────
            def _get_model_preds(model_key, run_fn):
                """Return predictions array. Load from cache or run and save."""
                cached_data = _load_preds(ctx_dir, ds_name, model_key)
                if cached_data is not None:
                    print(f"  {ds_name}/{model_key}: loaded from cache")
                    preds = cached_data["predictions"]
                    assert preds.shape[0] == n_samples, (
                        f"{ds_name}/{model_key}: cached predictions have "
                        f"{preds.shape[0]} samples but expected {n_samples}"
                    )
                    return preds
                preds = run_fn()
                _save_preds(
                    ctx_dir,
                    ds_name,
                    model_key,
                    history=hist_arr,
                    predictions=preds,
                    gt=gt_arr,
                    history_means=means_arr,
                    history_stds=stds_arr,
                )
                return preds

            # ── Migas-1.5 + Chronos (core) ───────────────────────────
            def _run_core():
                res = evaluate_migas_precomputed(
                    model,
                    loader=None,
                    device=args.device,
                    pred_len=args.pred_len,
                    prediction_key="migas15",
                    precomputed_summaries=summaries,
                    precomputed_historic=hist_eval,
                    precomputed_forecast=fcast_eval,
                    precomputed_means=ctx_means,
                    precomputed_stds=ctx_stds,
                    batch_size=args.batch_size,
                )
                return (
                    res["predictions"]["migas15"].numpy(),
                    res["predictions"]["timeseries"].numpy(),
                    res["gt"].numpy(),
                )

            if _has_preds(ctx_dir, ds_name, "migas15") and _has_preds(
                ctx_dir, ds_name, "chronos"
            ):
                _migas_cached = _load_preds(ctx_dir, ds_name, "migas15")
                _chronos_cached = _load_preds(ctx_dir, ds_name, "chronos")
                if _migas_cached is None or _chronos_cached is None:
                    print(f"  {ds_name}: cache corrupt, recomputing")
                    migas_preds, chronos_preds, gt = _run_core()
                    _save_preds(
                        ctx_dir, ds_name, "migas15",
                        hist_arr, migas_preds, gt, means_arr, stds_arr,
                    )
                    _save_preds(
                        ctx_dir, ds_name, "chronos",
                        hist_arr, chronos_preds, gt, means_arr, stds_arr,
                    )
                else:
                    migas_preds = _migas_cached["predictions"]
                    chronos_preds = _chronos_cached["predictions"]
                    gt = _migas_cached["gt"]
                    assert migas_preds.shape[0] == gt.shape[0], (
                        f"{ds_name}/migas15: predictions ({migas_preds.shape[0]}) "
                        f"vs gt ({gt.shape[0]}) sample count mismatch"
                    )
                    assert chronos_preds.shape[0] == gt.shape[0], (
                        f"{ds_name}/chronos: predictions ({chronos_preds.shape[0]}) "
                        f"vs gt ({gt.shape[0]}) sample count mismatch"
                    )
                    print(f"  {ds_name}: core cached")
            else:
                migas_preds, chronos_preds, gt = _run_core()
                _save_preds(
                    ctx_dir, ds_name, "migas15",
                    hist_arr, migas_preds, gt, means_arr, stds_arr,
                )
                _save_preds(
                    ctx_dir, ds_name, "chronos",
                    hist_arr, chronos_preds, gt, means_arr, stds_arr,
                )

            n_samples = gt.shape[0]
            migas_m = compute_metrics(migas_preds, gt)
            chro_m = compute_metrics(chronos_preds, gt)

            migas_per_window = np.mean(np.abs(migas_preds - gt), axis=1)
            chro_per_window = np.mean(np.abs(chronos_preds - gt), axis=1)
            windows_migas_better = int(np.sum(migas_per_window < chro_per_window))
            windows_chronos_better = int(np.sum(migas_per_window > chro_per_window))
            windows_tied = n_samples - windows_migas_better - windows_chronos_better

            row = {
                "dataset": ds_name,
                "n_samples": n_samples,
                "migas15_mean_mae": migas_m["mean_mae"],
                "migas15_median_mae": migas_m["median_mae"],
                "chronos_mean_mae": chro_m["mean_mae"],
                "chronos_median_mae": chro_m["median_mae"],
                "migas15_mean_mse": migas_m["mean_mse"],
                "chronos_mean_mse": chro_m["mean_mse"],
                "migas15_mean_mape": migas_m["mean_mape"],
                "migas15_median_mape": migas_m["median_mape"],
                "chronos_mean_mape": chro_m["mean_mape"],
                "chronos_median_mape": chro_m["median_mape"],
                "mae_improvement_pct": (
                    (chro_m["mean_mae"] - migas_m["mean_mae"])
                    / chro_m["mean_mae"]
                    * 100
                    if chro_m["mean_mae"] > 0
                    else 0.0
                ),
                "windows_migas15_better": windows_migas_better,
                "windows_chronos_better": windows_chronos_better,
                "windows_tied": windows_tied,
                "pct_windows_migas15_better": (
                    windows_migas_better / n_samples * 100
                    if n_samples > 0
                    else 0.0
                ),
            }

            # ── TimesFM baseline ─────────────────────────────────────
            if args.eval_timesfm:

                def _run_timesfm():
                    r = evaluate_timesfm_precomputed(
                        None,
                        args.device,
                        pred_len=args.pred_len,
                        precomputed_historic=hist_eval,
                        precomputed_forecast=fcast_eval,
                        precomputed_means=ctx_means,
                        precomputed_stds=ctx_stds,
                        batch_size=args.batch_size,
                    )
                    return r["predictions"]["timesfm_univar"].numpy()

                tfm_preds = _get_model_preds("timesfm", _run_timesfm)
                tfm_m = compute_metrics(tfm_preds, gt)
                row["timesfm_mean_mae"] = tfm_m["mean_mae"]
                row["timesfm_median_mae"] = tfm_m["median_mae"]
                row["timesfm_mean_mse"] = tfm_m["mean_mse"]
                row["timesfm_mean_mape"] = tfm_m["mean_mape"]
                row["timesfm_median_mape"] = tfm_m["median_mape"]
                row["migas15_vs_timesfm_improvement_pct"] = (
                    (tfm_m["mean_mae"] - migas_m["mean_mae"])
                    / tfm_m["mean_mae"]
                    * 100
                    if tfm_m["mean_mae"] > 0
                    else 0.0
                )

            # ── Toto baseline ────────────────────────────────────────
            if args.eval_toto:

                def _run_toto():
                    r = evaluate_toto_precomputed(
                        None,
                        args.device,
                        pred_len=args.pred_len,
                        precomputed_historic=hist_eval,
                        precomputed_forecast=fcast_eval,
                        precomputed_means=ctx_means,
                        precomputed_stds=ctx_stds,
                        batch_size=args.batch_size,
                    )
                    return r["predictions"]["toto_univar"].numpy()

                toto_preds = _get_model_preds("toto", _run_toto)
                toto_m = compute_metrics(toto_preds, gt)
                row["toto_mean_mae"] = toto_m["mean_mae"]
                row["toto_median_mae"] = toto_m["median_mae"]
                row["toto_mean_mse"] = toto_m["mean_mse"]
                row["toto_mean_mape"] = toto_m["mean_mape"]
                row["toto_median_mape"] = toto_m["median_mape"]
                row["migas15_vs_toto_improvement_pct"] = (
                    (toto_m["mean_mae"] - migas_m["mean_mae"])
                    / toto_m["mean_mae"]
                    * 100
                    if toto_m["mean_mae"] > 0
                    else 0.0
                )

            # ── TabPFN baseline ──────────────────────────────────────
            if args.eval_tabpfn:

                def _run_tabpfn():
                    r = evaluate_tabpfn_precomputed(
                        hist_eval,
                        fcast_eval,
                        args.pred_len,
                        batch_size=args.batch_size,
                        means=ctx_means,
                        stds=ctx_stds,
                    )
                    return r["predictions"]

                if _has_preds(ctx_dir, ds_name, "tabpfn"):
                    tabpfn_preds = _load_preds(ctx_dir, ds_name, "tabpfn")[
                        "predictions"
                    ]
                    assert tabpfn_preds.shape[0] == n_samples, (
                        f"{ds_name}/tabpfn: cached predictions have "
                        f"{tabpfn_preds.shape[0]} samples but expected {n_samples}"
                    )
                    print(f"  {ds_name}/tabpfn: loaded from cache")
                else:
                    tabpfn_preds = _run_tabpfn()
                    _save_preds(
                        ctx_dir,
                        ds_name,
                        "tabpfn",
                        hist_arr,
                        tabpfn_preds,
                        gt,
                        means_arr,
                        stds_arr,
                    )
                tabpfn_m = compute_metrics(tabpfn_preds, gt)
                row["tabpfn_mean_mae"] = tabpfn_m["mean_mae"]
                row["tabpfn_median_mae"] = tabpfn_m["median_mae"]
                row["tabpfn_mean_mse"] = tabpfn_m["mean_mse"]
                row["tabpfn_mean_mape"] = tabpfn_m["mean_mape"]
                row["tabpfn_median_mape"] = tabpfn_m["median_mape"]
                row["migas15_vs_tabpfn_improvement_pct"] = (
                    (tabpfn_m["mean_mae"] - migas_m["mean_mae"])
                    / tabpfn_m["mean_mae"]
                    * 100
                    if tabpfn_m["mean_mae"] > 0
                    else 0.0
                )

            # ── Prophet baseline ─────────────────────────────────────
            if args.eval_prophet:

                def _run_prophet():
                    r = evaluate_prophet_precomputed(
                        hist_eval,
                        fcast_eval,
                        args.pred_len,
                        means=ctx_means,
                        stds=ctx_stds,
                    )
                    return r["predictions"]

                prophet_preds = _get_model_preds("prophet", _run_prophet)
                prophet_m = compute_metrics(prophet_preds, gt)
                row["prophet_mean_mae"] = prophet_m["mean_mae"]
                row["prophet_median_mae"] = prophet_m["median_mae"]
                row["prophet_mean_mse"] = prophet_m["mean_mse"]
                row["prophet_mean_mape"] = prophet_m["mean_mape"]
                row["prophet_median_mape"] = prophet_m["median_mape"]
                row["migas15_vs_prophet_improvement_pct"] = (
                    (prophet_m["mean_mae"] - migas_m["mean_mae"])
                    / prophet_m["mean_mae"]
                    * 100
                    if prophet_m["mean_mae"] > 0
                    else 0.0
                )

            # ── SARIMA baseline ──────────────────────────────────────
            if args.eval_sarima:

                def _run_sarima():
                    r = evaluate_sarima_precomputed(
                        hist_eval, fcast_eval, args.pred_len
                    )
                    return r["predictions"]

                sarima_preds = _get_model_preds("sarima", _run_sarima)
                sarima_m = compute_metrics(sarima_preds, gt)
                row["sarima_mean_mae"] = sarima_m["mean_mae"]
                row["sarima_median_mae"] = sarima_m["median_mae"]
                row["sarima_mean_mse"] = sarima_m["mean_mse"]
                row["sarima_mean_mape"] = sarima_m["mean_mape"]
                row["sarima_median_mape"] = sarima_m["median_mape"]
                row["migas15_vs_sarima_improvement_pct"] = (
                    (sarima_m["mean_mae"] - migas_m["mean_mae"])
                    / sarima_m["mean_mae"]
                    * 100
                    if sarima_m["mean_mae"] > 0
                    else 0.0
                )

            rows.append(row)

            r_migas = float(row["migas15_mean_mae"])
            r_chro = float(row["chronos_mean_mae"])
            if r_migas < r_chro:
                migas_wins += 1
                winner = "Migas-1.5"
            elif r_migas > r_chro:
                chronos_wins += 1
                winner = "Chronos"
            else:
                ties += 1
                winner = "Tie"

            parts = [
                f"  {ds_name:30s}  n={n_samples:4d}",
                f"Migas-1.5={r_migas:.4f}",
                f"Chronos={r_chro:.4f}",
            ]
            if args.eval_timesfm and "timesfm_mean_mae" in row:
                parts.append(f"TimesFM={float(row['timesfm_mean_mae']):.4f}")
            if args.eval_toto and "toto_mean_mae" in row:
                parts.append(f"Toto={float(row['toto_mean_mae']):.4f}")
            if args.eval_tabpfn and "tabpfn_mean_mae" in row:
                parts.append(f"TabPFN={float(row['tabpfn_mean_mae']):.4f}")
            if args.eval_prophet and "prophet_mean_mae" in row:
                parts.append(f"Prophet={float(row['prophet_mean_mae']):.4f}")
            if args.eval_sarima and "sarima_mean_mae" in row:
                parts.append(f"SARIMA={float(row['sarima_mean_mae']):.4f}")
            parts.append(f"Impr={float(row['mae_improvement_pct']):+.1f}%")
            parts.append(f"[{winner}]")
            print("  ".join(parts))

        # ── Summary for this context length ──────────────────────────
        n_datasets = len(rows)
        print("\n" + "=" * 80)
        print(f"SUMMARY  (context_length = {ctx_len})")
        print("=" * 80)
        print(f"Datasets evaluated: {n_datasets}")
        print(f"Migas-1.5 wins (vs Chronos): {migas_wins}/{n_datasets}")
        print(f"Chronos wins:                {chronos_wins}/{n_datasets}")
        if ties:
            print(f"Ties:                        {ties}/{n_datasets}")

        if rows:
            metrics = [
                "mean_mae",
                "median_mae",
                "mean_mse",
                "mean_mape",
                "median_mape",
            ]
            metric_labels = {
                "mean_mae": "Mean MAE",
                "median_mae": "Median MAE",
                "mean_mse": "Mean MSE",
                "mean_mape": "Mean MAPE",
                "median_mape": "Median MAPE",
            }

            models_list = [
                ("Migas-1.5 (Ours)", "migas15"),
                ("Chronos", "chronos"),
            ]
            _optional_models = [
                (args.eval_timesfm, "TimesFM", "timesfm"),
                (args.eval_toto, "Toto", "toto"),
                (args.eval_tabpfn, "TabPFN", "tabpfn"),
                (args.eval_prophet, "Prophet", "prophet"),
                (args.eval_sarima, "SARIMA", "sarima"),
            ]
            for _flag, _lbl, _pfx in _optional_models:
                if _flag and all(f"{_pfx}_mean_mae" in r for r in rows):
                    models_list.append((_lbl, _pfx))

            n_samples_arr = np.array([r["n_samples"] for r in rows])
            weights = n_samples_arr / n_samples_arr.sum()

            table_rows = []
            for model_label, prefix in models_list:
                for metric in metrics:
                    col = f"{prefix}_{metric}"
                    vals = np.array(
                        [r.get(col, float("nan")) for r in rows]
                    )
                    normal_mean = float(np.mean(vals))
                    weighted_mean = float(np.sum(vals * weights))
                    table_rows.append(
                        [
                            model_label,
                            metric_labels[metric],
                            f"{normal_mean:.6f}",
                            f"{weighted_mean:.6f}",
                        ]
                    )

            headers = ["Model", "Metric", "Mean", "Weighted Mean"]
            print(
                f"\n  Aggregate metrics across {n_datasets} datasets "
                f"(weighted by n_samples, total={int(n_samples_arr.sum())})\n"
            )
            print(tabulate(table_rows, headers=headers, tablefmt="grid"))

            all_impr = [r["mae_improvement_pct"] for r in rows]
            total_windows = sum(r["n_samples"] for r in rows)
            total_migas_better = sum(
                r["windows_migas15_better"] for r in rows
            )
            total_chronos_better = sum(
                r["windows_chronos_better"] for r in rows
            )
            total_tied = sum(r["windows_tied"] for r in rows)
            avg_pct = np.mean(
                [r["pct_windows_migas15_better"] for r in rows]
            )

            print(
                "\nPer-window stats -- Migas-1.5 vs Chronos "
                "(across all datasets):"
            )
            print(f"  Total windows:       {total_windows}")
            print(
                f"  Migas-1.5 better:    {total_migas_better}/{total_windows} "
                f"({total_migas_better / total_windows * 100:.1f}%)"
            )
            print(
                f"  Chronos better:      {total_chronos_better}/{total_windows} "
                f"({total_chronos_better / total_windows * 100:.1f}%)"
            )
            if total_tied:
                print(
                    f"  Tied:                {total_tied}/{total_windows} "
                    f"({total_tied / total_windows * 100:.1f}%)"
                )
            print(
                f"  Avg % windows Migas-1.5 better (per dataset): "
                f"{avg_pct:.1f}%"
            )

            print(
                f"\n  Migas-1.5 vs Chronos improvement: "
                f"{np.mean(all_impr):+.2f}%"
            )
            for _flag, _lbl, _pfx in _optional_models:
                _impr_key = f"migas15_vs_{_pfx}_improvement_pct"
                _mae_key = f"{_pfx}_mean_mae"
                if _flag and all(_impr_key in r for r in rows):
                    _impr_vals = [r[_impr_key] for r in rows]
                    print(
                        f"  Migas-1.5 vs {_lbl} improvement: "
                        f"{np.mean(_impr_vals):+.2f}%"
                    )
                    _beats = sum(
                        1
                        for r in rows
                        if r["migas15_mean_mae"] < r[_mae_key]
                    )
                    print(f"  Migas-1.5 beats {_lbl}: {_beats}/{n_datasets}")

        # ── Save CSV for this context length ─────────────────────────
        if rows:
            all_keys = {}
            for r in rows:
                for k in r:
                    all_keys[k] = True
            fieldnames = list(all_keys.keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with open(csv_path, "a") as f:
                f.write(
                    f"\n# Migas-1.5 wins: {migas_wins}/{n_datasets}  "
                    f"Chronos wins: {chronos_wins}/{n_datasets}  "
                    f"Avg improvement: {np.mean(all_impr):+.2f}%\n"
                )

            print(f"\nResults saved to {csv_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
