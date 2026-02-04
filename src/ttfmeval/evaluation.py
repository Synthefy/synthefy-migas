#!/usr/bin/env python3
"""
Evaluate forecasting models on CSV datasets.
Runs TTFM and baselines, computes MSE/MAE/MAPE and directional accuracy.
Fully standalone: no dependency on the training repo.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ttfmeval.baselines import BASELINE_REGISTRY, get_baseline_for_prediction_key
from ttfmeval.dataset import (
    LateFusionDataset,
    collate_fn as late_fusion_collate,
    list_csv_files,
)
from ttfmeval.model import build_model


# =============================================================================
# METRICS
# =============================================================================


def get_mean_and_median_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mse = (pred - gt) ** 2
    mae = np.abs(pred - gt)
    mape = np.abs((pred - gt) / (np.abs(gt) + 1e-8))
    mse_per_sample = np.mean(mse, axis=1)
    mae_per_sample = np.mean(mae, axis=1)
    mape_per_sample = np.mean(mape, axis=1)
    return {
        "mean_mse": np.mean(mse_per_sample),
        "mean_mae": np.mean(mae_per_sample),
        "mean_mape": np.mean(mape_per_sample),
        "median_mse": np.median(mse_per_sample),
        "median_mae": np.median(mae_per_sample),
        "median_mape": np.median(mape_per_sample),
    }


def compute_unscaled_mape(
    pred: np.ndarray,
    gt: np.ndarray,
    history_means: np.ndarray,
    history_stds: np.ndarray,
) -> np.ndarray:
    if pred.ndim == 2:
        history_mean = history_means[:, np.newaxis]
        history_std = history_stds[:, np.newaxis]
    else:
        history_mean = history_means[:, np.newaxis, np.newaxis]
        history_std = history_stds[:, np.newaxis, np.newaxis]
    pred_unscaled = pred * history_std + history_mean
    gt_unscaled = gt * history_std + history_mean
    mape = np.abs((pred_unscaled - gt_unscaled) / (np.abs(gt_unscaled) + 1e-8))
    if mape.ndim == 2:
        mape_per_sample = np.mean(mape, axis=1)
    else:
        mape_per_sample = np.mean(mape, axis=(1, 2))
    return mape_per_sample * 100


def compute_directional_accuracy(
    input_arr: np.ndarray, pred_arr: np.ndarray, gt_arr: np.ndarray, step: int
) -> float:
    if input_arr.ndim == 3:
        input_arr = input_arr[..., 0]
    if pred_arr.ndim == 3:
        pred_arr = pred_arr[..., 0]
    if gt_arr.ndim == 3:
        gt_arr = gt_arr[..., 0]
    last_input = input_arr[:, -1]
    step_idx = step - 1
    if step_idx >= pred_arr.shape[1]:
        return float("nan")
    pred_direction = np.sign(pred_arr[:, step_idx] - last_input)
    gt_direction = np.sign(gt_arr[:, step_idx] - last_input)
    valid_mask = gt_direction != 0
    if valid_mask.sum() == 0:
        return 0.0
    correct = (pred_direction[valid_mask] == gt_direction[valid_mask]).sum()
    return correct / valid_mask.sum() * 100


def compute_step_to_step_directional_accuracy(
    pred_arr: np.ndarray, gt_arr: np.ndarray, step: int
) -> float:
    if step < 2:
        return float("nan")
    if pred_arr.ndim == 3:
        pred_arr = pred_arr[..., 0]
    if gt_arr.ndim == 3:
        gt_arr = gt_arr[..., 0]
    step_idx = step - 1
    prev_step_idx = step - 2
    if step_idx >= pred_arr.shape[1]:
        return float("nan")
    pred_direction = np.sign(pred_arr[:, step_idx] - pred_arr[:, prev_step_idx])
    gt_direction = np.sign(gt_arr[:, step_idx] - gt_arr[:, prev_step_idx])
    valid_mask = gt_direction != 0
    if valid_mask.sum() == 0:
        return 0.0
    correct = (pred_direction[valid_mask] == gt_direction[valid_mask]).sum()
    return correct / valid_mask.sum() * 100


def compute_all_metrics(results: dict, pred_len: int = 4) -> dict:
    input_arr = results["input"].numpy()
    gt_arr = results["gt"].numpy()
    all_metrics = {}
    for model_name, pred_tensor in results["predictions"].items():
        pred_arr = pred_tensor.numpy()
        scaled_metrics = get_mean_and_median_metrics(pred_arr, gt_arr)
        dir_acc = {}
        step_dir_acc = {}
        for step in range(1, pred_len + 1):
            dir_acc[step] = compute_directional_accuracy(
                input_arr, pred_arr, gt_arr, step
            )
            step_dir_acc[step] = compute_step_to_step_directional_accuracy(
                pred_arr, gt_arr, step
            )
        all_metrics[model_name] = {
            "scaled": scaled_metrics,
            "directional_acc": dir_acc,
            "step_directional_acc": step_dir_acc,
        }
    return all_metrics


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate TTFM and baselines on CSV datasets (standalone)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seq_len", type=int, default=64, help="Context length")
    p.add_argument("--pred_len", type=int, default=16, help="Prediction horizon")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--val_length", type=int, default=1000)
    p.add_argument(
        "--datasets_dir",
        type=str,
        default=os.environ.get("TTFM_EVAL_DATASETS_DIR", "./data/test"),
        help="Directory containing CSV files (t, y_t, text)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("TTFM_EVAL_OUTPUT_DIR", "./results"),
        help="Output directory for predictions and metrics",
    )
    p.add_argument("--device", type=str, default="cuda")

    default_checkpoint = os.environ.get("TTFM_CHECKPOINT", "")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help="Path to TTFM checkpoint (or set TTFM_CHECKPOINT)",
    )
    p.add_argument("--checkpoint_timesfm", type=str, default="")

    p.add_argument("--univariate_model", type=str, default="chronos")
    p.add_argument("--text_embedder", type=str, default="finbert")
    p.add_argument("--use_timestamps", action="store_true")
    p.add_argument("--llm_base_url", type=str, default="http://localhost:8004/v1")
    p.add_argument("--llm_model", type=str, default="openai/gpt-oss-120b")
    p.add_argument("--gpt_cov_noise_std", type=float, default=0.05)
    p.add_argument("--prophet_freq", type=str, default="D")

    for name, config in BASELINE_REGISTRY.items():
        p.add_argument(
            f"--eval_{name}",
            action="store_true",
            help=config.help_text,
        )
    return p.parse_args()


# =============================================================================
# FILE / CACHE HELPERS
# =============================================================================


def extract_dataset_name(csv_path: str) -> str:
    return os.path.splitext(os.path.basename(csv_path))[0]


def get_dataset_output_dir(output_dir: str, dataset_name: str) -> str:
    dataset_output_dir = os.path.join(output_dir, "outputs", dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    return dataset_output_dir


def load_per_dataset_results(output_dir: str, dataset_name: str) -> dict:
    dataset_dir = get_dataset_output_dir(output_dir, dataset_name)
    input_path = os.path.join(dataset_dir, "input.npy")
    gt_path = os.path.join(dataset_dir, "gt.npy")
    if not os.path.exists(input_path) or not os.path.exists(gt_path):
        return None
    results = {
        "input": torch.from_numpy(np.load(input_path)),
        "gt": torch.from_numpy(np.load(gt_path)),
        "predictions": {},
    }
    for file in os.listdir(dataset_dir):
        if file.endswith("_pred.npy"):
            model_name = file[:-9]
            results["predictions"][model_name] = torch.from_numpy(
                np.load(os.path.join(dataset_dir, file))
            )
    return results if results["predictions"] else None


def save_per_dataset_results(output_dir: str, dataset_name: str, results: dict):
    dataset_dir = get_dataset_output_dir(output_dir, dataset_name)
    np.save(os.path.join(dataset_dir, "input.npy"), results["input"].numpy())
    np.save(os.path.join(dataset_dir, "gt.npy"), results["gt"].numpy())
    for model_name, pred_tensor in results["predictions"].items():
        np.save(
            os.path.join(dataset_dir, f"{model_name}_pred.npy"),
            pred_tensor.numpy(),
        )


def get_enabled_baselines(args) -> list:
    enabled = []
    for name in BASELINE_REGISTRY:
        if getattr(args, f"eval_{name}", False):
            enabled.append(name)
    return enabled


def get_expected_prediction_keys(args) -> list:
    keys = []
    for name in get_enabled_baselines(args):
        keys.extend(BASELINE_REGISTRY[name].prediction_keys)
    return list(set(keys))


def check_cache_status(
    output_dir: str, dataset_name: str, expected_n_samples: int, args
) -> tuple:
    dataset_dir = os.path.join(output_dir, "outputs", dataset_name)
    if not os.path.exists(dataset_dir):
        return False, []
    input_path = os.path.join(dataset_dir, "input.npy")
    if not os.path.exists(input_path):
        return False, []
    input_array = np.load(input_path)
    if input_array.shape[0] != expected_n_samples:
        return False, []
    expected_keys = get_expected_prediction_keys(args)
    missing_keys = []
    for key in expected_keys:
        if not os.path.exists(os.path.join(dataset_dir, f"{key}_pred.npy")):
            missing_keys.append(key)
    return len(missing_keys) == 0, missing_keys


def get_expected_sample_count(csv_path: str, args, LateFusionDataset) -> int:
    dataset = LateFusionDataset(
        args.seq_len + args.pred_len,
        args.pred_len,
        [csv_path],
        split="test",
        val_length=1000,
        stride=1,
    )
    return len(dataset)


def compute_raw_mean_std_for_dataset(
    csv_path: str, args, LateFusionDataset, late_fusion_collate
) -> tuple:
    dataset = LateFusionDataset(
        args.seq_len + args.pred_len,
        args.pred_len,
        [csv_path],
        split="test",
        val_length=1000,
        stride=1,
    )
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=late_fusion_collate
    )
    history_means = []
    history_stds = []
    for batch_dict in dataloader:
        history_means.extend(batch_dict["history_means"])
        history_stds.extend(batch_dict["history_stds"])
    return np.array(history_means), np.array(history_stds)


# =============================================================================
# SINGLE-DATASET EVALUATION
# =============================================================================


def evaluate_single_dataset(
    csv_path: str,
    args,
    LateFusionDataset,
    late_fusion_collate,
    models: dict,
    baselines_to_eval: list,
    precomputed_results: dict = None,
):
    dataset = LateFusionDataset(
        args.seq_len + args.pred_len,
        args.pred_len,
        [csv_path],
        split="test",
        val_length=1000,
        stride=1,
    )
    if len(dataset) == 0:
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=late_fusion_collate,
    )

    all_results = precomputed_results.copy() if precomputed_results else {}
    if "predictions" not in all_results:
        all_results["predictions"] = {}

    univariate_model = "chronos"
    if getattr(args, "univariate_model", None):
        name = str(args.univariate_model).lower()
        if "timesfm" in name:
            univariate_model = "timesfm"
        elif "prophet" in name:
            univariate_model = "prophet"

    for baseline_name in baselines_to_eval:
        config = BASELINE_REGISTRY[baseline_name]
        kwargs = {"pred_len": args.pred_len}
        for func_arg, args_attr in config.extra_args_map.items():
            kwargs[func_arg] = getattr(args, args_attr)

        if baseline_name == "ttfmlf":
            model = models.get("model")
            result = config.eval_func(
                model,
                dataloader,
                args.device,
                args.pred_len,
                model_type="ttfmlf",
                prediction_key="ttfm",
                use_timestamps=args.use_timestamps,
                univariate_model=univariate_model,
            )
        elif baseline_name == "ttfmlf_timesfm":
            model = models.get("model_timesfm")
            result = config.eval_func(
                model,
                dataloader,
                args.device,
                args.pred_len,
                model_type="ttfmlf",
                prediction_key="ttfm_timesfm",
                use_timestamps=args.use_timestamps,
                univariate_model="timesfm",
            )
        elif baseline_name == "chronos2":
            result = config.eval_func(
                dataloader, args.device, args.pred_len, eval_multivar=False
            )
        elif baseline_name == "chronos2_multivar":
            result = config.eval_func(
                dataloader, args.device, args.pred_len, eval_multivar=True
            )
        elif baseline_name == "chronos2_gpt":
            gpt_forecasts = None
            if "gpt_forecast" in all_results["predictions"]:
                gpt_forecasts = all_results["predictions"]["gpt_forecast"].numpy()
            else:
                dataset_name = extract_dataset_name(csv_path)
                dataset_dir = get_dataset_output_dir(args.output_dir, dataset_name)
                gpt_path = os.path.join(dataset_dir, "gpt_forecast_pred.npy")
                if os.path.exists(gpt_path):
                    gpt_forecasts = np.load(gpt_path)
            if gpt_forecasts is None:
                print(
                    "  Warning: gpt_forecast not available for chronos2_gpt, skipping"
                )
                continue
            result = config.eval_func(
                dataloader,
                args.device,
                args.pred_len,
                precomputed_gpt_forecasts=gpt_forecasts,
                **kwargs,
            )
        else:
            result = config.eval_func(dataloader, args.device, **kwargs)

        if result and "gt" not in all_results:
            all_results["input"] = result["input"]
            all_results["gt"] = result["gt"]
        if result:
            all_results["predictions"].update(result["predictions"])

    if "gt" not in all_results or all_results["gt"].shape[0] == 0:
        return None
    return all_results


def flatten_metrics_for_csv(dataset_name: str, metrics: dict, pred_len: int) -> dict:
    row = {"dataset_name": dataset_name}
    for model_name, model_metrics in metrics.items():
        for key in [
            "mean_mse",
            "median_mse",
            "mean_mae",
            "median_mae",
            "mean_mape",
            "median_mape",
        ]:
            row[f"{model_name}_scaled_{key}"] = model_metrics["scaled"][key]
        for step in range(1, pred_len + 1):
            row[f"{model_name}_dir_acc_step{step}"] = model_metrics["directional_acc"][
                step
            ]
    if "timeseries" in metrics and "ttfm" in metrics:
        ts_mae = metrics["timeseries"]["scaled"]["mean_mae"]
        ttfm_mae = metrics["ttfm"]["scaled"]["mean_mae"]
        row["mae_improvement_pct"] = (
            ((ts_mae - ttfm_mae) / ts_mae * 100) if ts_mae > 0 else 0
        )
    return row


# =============================================================================
# OVERALL STATS
# =============================================================================


def get_sample_indices_after_date(
    csv_path: str, window_len: int, cutoff_date: str = "2024-06-01"
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df["t_date"] = pd.to_datetime(df["t"], errors="coerce")
    df = df[df["t_date"].notna()]
    cutoff = pd.to_datetime(cutoff_date)
    valid_indices = []
    for i in range(len(df) - window_len + 1):
        if df.iloc[i]["t_date"] >= cutoff:
            valid_indices.append(i)
    return np.array(valid_indices)


def generate_overall_stats_csv(
    output_dir: str, args, LateFusionDataset, late_fusion_collate
):
    dataset_csvs = list_csv_files(args.datasets_dir)
    model_order = []
    for config in BASELINE_REGISTRY.values():
        model_order.extend(config.prediction_keys)
    model_order = list(dict.fromkeys(model_order))

    overall_stats = []
    for csv_path in dataset_csvs:
        dataset_name = extract_dataset_name(csv_path)
        dataset_results = load_per_dataset_results(output_dir, dataset_name)
        if dataset_results is None:
            continue
        n_samples = dataset_results["gt"].shape[0]
        history_means, history_stds = compute_raw_mean_std_for_dataset(
            csv_path, args, LateFusionDataset, late_fusion_collate
        )

        row_data = {
            "dataset_name": dataset_name,
            "n_samples": args.seq_len + n_samples + args.pred_len - 1,
            "n_eval_samples": n_samples,
        }
        mae_dict = {}
        gt_np = dataset_results["gt"].numpy()
        for model_name, pred_tensor in dataset_results["predictions"].items():
            pred = pred_tensor.numpy()
            mae = np.mean(np.abs(pred - gt_np), axis=1)
            mae_dict[model_name] = mae
            row_data[f"{model_name}_mean_mae"] = np.mean(mae)
            row_data[f"{model_name}_median_mae"] = np.median(mae)
            row_data[f"{model_name}_std_mae"] = np.std(mae)
            mse = np.mean((pred - gt_np) ** 2, axis=1)
            row_data[f"{model_name}_mean_mse"] = np.mean(mse)
            row_data[f"{model_name}_median_mse"] = np.median(mse)
            mape = compute_unscaled_mape(pred, gt_np, history_means, history_stds)
            row_data[f"{model_name}_mean_mape"] = np.mean(mape)
            row_data[f"{model_name}_median_mape"] = np.median(mape)
            row_data[f"{model_name}_std_mape"] = np.std(mape)
        if "ttfm" in mae_dict and "timeseries" in mae_dict:
            ttfm_wins = np.sum(mae_dict["ttfm"] < mae_dict["timeseries"])
            row_data["ttfm_win_pct"] = (ttfm_wins / n_samples) * 100
            gap_mean = row_data["timeseries_mean_mae"] - row_data["ttfm_mean_mae"]
            gap_median = row_data["timeseries_median_mae"] - row_data["ttfm_median_mae"]
            row_data["gap_mean"] = gap_mean
            row_data["gap_median"] = gap_median
            ts_mean = row_data["timeseries_mean_mae"]
            ts_median = row_data["timeseries_median_mae"]
            row_data["improvement_pct_mean"] = (
                (gap_mean / ts_mean * 100) if ts_mean > 0 else 0
            )
            row_data["improvement_pct_median"] = (
                (gap_median / ts_median * 100) if ts_median > 0 else 0
            )
        overall_stats.append(row_data)

    if not overall_stats:
        return
    overall_df = pd.DataFrame(overall_stats)
    ordered_cols = [
        "dataset_name",
        "improvement_pct_mean",
        "n_samples",
        "n_eval_samples",
    ]
    for suffix in [
        "median_mae",
        "mean_mae",
        "median_mape",
        "mean_mape",
        "std_mae",
        "std_mape",
        "median_mse",
        "mean_mse",
    ]:
        for model in model_order:
            col = f"{model}_{suffix}"
            if col in overall_df.columns:
                ordered_cols.append(col)
    for col in [
        "ttfm_win_pct",
        "gap_median",
        "gap_mean",
        "improvement_pct_mean",
        "improvement_pct_median",
    ]:
        if col in overall_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    overall_df = overall_df[[c for c in ordered_cols if c in overall_df.columns]]
    overall_df.to_csv(
        os.path.join(output_dir, f"stats_Context_{args.seq_len}_allsamples.csv"),
        index=False,
    )

    # June 2024+ filtered stats
    overall_stats_f = []
    for csv_path in dataset_csvs:
        dataset_name = extract_dataset_name(csv_path)
        dataset_results = load_per_dataset_results(output_dir, dataset_name)
        if dataset_results is None:
            continue
        filtered_indices = get_sample_indices_after_date(
            csv_path, args.seq_len + args.pred_len, "2024-06-01"
        )
        if len(filtered_indices) == 0:
            continue
        gt_np = dataset_results["gt"].numpy()
        filtered_indices = filtered_indices[filtered_indices < gt_np.shape[0]]
        if len(filtered_indices) == 0:
            continue
        gt_np_f = gt_np[filtered_indices]
        n_f = len(filtered_indices)
        history_means, history_stds = compute_raw_mean_std_for_dataset(
            csv_path, args, LateFusionDataset, late_fusion_collate
        )
        history_means = history_means[filtered_indices]
        history_stds = history_stds[filtered_indices]
        row_data = {
            "dataset_name": dataset_name,
            "n_samples": args.seq_len + n_f + args.pred_len - 1,
            "n_eval_samples": n_f,
        }
        for model_name, pred_tensor in dataset_results["predictions"].items():
            pred = pred_tensor.numpy()[filtered_indices]
            mae = np.mean(np.abs(pred - gt_np_f), axis=1)
            row_data[f"{model_name}_mean_mae"] = np.mean(mae)
            row_data[f"{model_name}_median_mae"] = np.median(mae)
            row_data[f"{model_name}_std_mae"] = np.std(mae)
            mse = np.mean((pred - gt_np_f) ** 2, axis=1)
            row_data[f"{model_name}_mean_mse"] = np.mean(mse)
            row_data[f"{model_name}_median_mse"] = np.median(mse)
            mape = compute_unscaled_mape(pred, gt_np_f, history_means, history_stds)
            row_data[f"{model_name}_mean_mape"] = np.mean(mape)
            row_data[f"{model_name}_median_mape"] = np.median(mape)
            row_data[f"{model_name}_std_mape"] = np.std(mape)
        if "ttfm_mean_mae" in row_data and "timeseries_mean_mae" in row_data:
            ttfm_pred = dataset_results["predictions"]["ttfm"].numpy()[filtered_indices]
            ts_pred = dataset_results["predictions"]["timeseries"].numpy()[
                filtered_indices
            ]
            mae_ttfm = np.mean(np.abs(ttfm_pred - gt_np_f), axis=1)
            mae_ts = np.mean(np.abs(ts_pred - gt_np_f), axis=1)
            row_data["ttfm_win_pct"] = (np.sum(mae_ttfm < mae_ts) / n_f) * 100
            gap_mean = row_data["timeseries_mean_mae"] - row_data["ttfm_mean_mae"]
            gap_median = row_data["timeseries_median_mae"] - row_data["ttfm_median_mae"]
            row_data["gap_mean"] = gap_mean
            row_data["gap_median"] = gap_median
            ts_mean = row_data["timeseries_mean_mae"]
            ts_median = row_data["timeseries_median_mae"]
            row_data["improvement_pct_mean"] = (
                (gap_mean / ts_mean * 100) if ts_mean > 0 else 0
            )
            row_data["improvement_pct_median"] = (
                (gap_median / ts_median * 100) if ts_median > 0 else 0
            )
        overall_stats_f.append(row_data)
    if overall_stats_f:
        filtered_df = pd.DataFrame(overall_stats_f)
        filtered_df = filtered_df[[c for c in ordered_cols if c in filtered_df.columns]]
        filtered_df.to_csv(
            os.path.join(output_dir, f"stats_Context_{args.seq_len}_june2024plus.csv"),
            index=False,
        )


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_models(args) -> dict:
    models = {}
    enabled = get_enabled_baselines(args)

    if "ttfmlf" in enabled:
        if not args.checkpoint or not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(
                "TTFM checkpoint required for --eval_ttfmlf. "
                "Set --checkpoint /path/to/model.pt or TTFM_CHECKPOINT env var."
            )
        print("\nLoading TTFM model...")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = build_model(
            pred_len=16,
            device=args.device,
            chronos_device=args.device,
            text_embedder=args.text_embedder,
            text_embedder_device=args.device,
            use_separate_summary_embedders=True,
            use_multiple_horizon_embedders=True,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(args.device)
        models["model"] = model
        print("TTFM model loaded.")

    if "ttfmlf_timesfm" in enabled:
        if not args.checkpoint_timesfm or not os.path.isfile(args.checkpoint_timesfm):
            raise FileNotFoundError(
                "TTFM-TimesFM checkpoint required for --eval_ttfmlf_timesfm. Set --checkpoint_timesfm."
            )
        print("\nLoading TTFM-TimesFM model...")
        checkpoint = torch.load(args.checkpoint_timesfm, map_location=args.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = build_model(
            pred_len=16,
            device=args.device,
            chronos_device=args.device,
            text_embedder=args.text_embedder,
            text_embedder_device=args.device,
            use_separate_summary_embedders=True,
            use_multiple_horizon_embedders=True,
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(args.device)
        models["model_timesfm"] = model
        print("TTFM-TimesFM model loaded.")

    return models


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    datasets_dir_name = os.path.basename(os.path.normpath(args.datasets_dir))
    output_dir = os.path.join(
        args.output_dir, datasets_dir_name, f"context_{args.seq_len}"
    )
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    dataset_csvs = list_csv_files(args.datasets_dir)
    enabled_baselines = get_enabled_baselines(args)

    if not enabled_baselines:
        print(
            "No baselines enabled. Use --eval_<name> (e.g. --eval_ttfmlf --eval_chronos2)."
        )
        return

    print(f"\nEnabled baselines: {', '.join(enabled_baselines)}")
    print(f"Expected prediction keys: {get_expected_prediction_keys(args)}")

    models = load_models(args)

    print("\n" + "=" * 80)
    print("EVALUATION PLAN")
    print("=" * 80)
    evaluation_plan = []
    for csv_path in dataset_csvs:
        dataset_name = extract_dataset_name(csv_path)
        expected_n_samples = get_expected_sample_count(
            csv_path, args, LateFusionDataset
        )
        if expected_n_samples <= 0:
            continue
        use_cache, missing_keys = check_cache_status(
            output_dir, dataset_name, expected_n_samples, args
        )
        if use_cache:
            evaluation_plan.append(
                {
                    "dataset": dataset_name,
                    "samples": expected_n_samples,
                    "status": "cached",
                }
            )
        elif missing_keys:
            baselines_to_run = set()
            for key in missing_keys:
                baseline = get_baseline_for_prediction_key(key)
                if baseline and baseline in enabled_baselines:
                    baselines_to_run.add(baseline)
            evaluation_plan.append(
                {
                    "dataset": dataset_name,
                    "samples": expected_n_samples,
                    "status": "partial",
                    "to_eval": list(baselines_to_run),
                }
            )
        else:
            evaluation_plan.append(
                {
                    "dataset": dataset_name,
                    "samples": expected_n_samples,
                    "status": "full",
                    "to_eval": enabled_baselines,
                }
            )

    for plan in evaluation_plan:
        print(f"\n{plan['dataset']}: {plan['samples']} samples", end="")
        if plan["status"] == "cached":
            print(" [cached]")
        else:
            print(f" -> evaluate: {', '.join(plan.get('to_eval', []))}")

    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80 + "\n")

    stats = {"successful": 0, "cached": 0, "failed": 0, "skipped": 0}

    for csv_path in tqdm(dataset_csvs, desc="Datasets"):
        dataset_name = extract_dataset_name(csv_path)
        expected_n_samples = get_expected_sample_count(
            csv_path, args, LateFusionDataset
        )
        if expected_n_samples <= 0:
            stats["skipped"] += 1
            continue

        use_cache, missing_keys = check_cache_status(
            output_dir, dataset_name, expected_n_samples, args
        )

        if use_cache:
            cached_results = load_per_dataset_results(output_dir, dataset_name)
            if cached_results:
                metrics = compute_all_metrics(cached_results, args.pred_len)
                stats["cached"] += 1
                print(
                    f"\n{dataset_name} (cached): {cached_results['gt'].shape[0]} samples"
                )
                for model_name in sorted(cached_results["predictions"].keys()):
                    if model_name in metrics:
                        print(
                            f"  {model_name}: MAE={metrics[model_name]['scaled']['mean_mae']:.4f}"
                        )
                continue

        if missing_keys:
            baselines_to_eval = []
            for key in missing_keys:
                baseline = get_baseline_for_prediction_key(key)
                if baseline and baseline in enabled_baselines:
                    baselines_to_eval.append(baseline)
            baselines_to_eval = list(set(baselines_to_eval))
            for baseline in baselines_to_eval.copy():
                dep = BASELINE_REGISTRY[baseline].depends_on
                if dep and dep not in baselines_to_eval and dep in enabled_baselines:
                    baselines_to_eval.insert(0, dep)
        else:
            baselines_to_eval = enabled_baselines.copy()

        cached_results = (
            load_per_dataset_results(output_dir, dataset_name) if missing_keys else None
        )

        try:
            results = evaluate_single_dataset(
                csv_path,
                args,
                LateFusionDataset,
                late_fusion_collate,
                models,
                baselines_to_eval,
                precomputed_results=cached_results,
            )
        except Exception as e:
            print(f"\n{dataset_name} failed: {e}")
            stats["failed"] += 1
            continue

        if results is None:
            stats["failed"] += 1
            continue

        stats["successful"] += 1
        save_per_dataset_results(output_dir, dataset_name, results)
        metrics = compute_all_metrics(results, args.pred_len)
        print(f"\n{dataset_name}: {results['gt'].shape[0]} samples")
        for model_name in sorted(results["predictions"].keys()):
            if model_name in metrics:
                print(
                    f"  {model_name}: MAE={metrics[model_name]['scaled']['mean_mae']:.4f}"
                )

    generate_overall_stats_csv(output_dir, args, LateFusionDataset, late_fusion_collate)

    print(
        f"\nDone: {stats['successful']} evaluated, {stats['cached']} cached, "
        f"{stats['failed']} failed, {stats['skipped']} skipped."
    )
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
