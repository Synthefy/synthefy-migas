#!/usr/bin/env python3
"""
Evaluate forecasting models on CSV datasets.
Runs TTFM and baselines, computes MSE/MAE/MAPE and directional accuracy.
Fully standalone: no dependency on the training repo.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ttfmeval.baselines import BASELINE_REGISTRY, get_baseline_for_prediction_key
from ttfmeval.dataset import (
    LateFusionDataset,
    collate_fn as late_fusion_collate,
    get_datasets_dir_from_hf,
    list_csv_files,
)
from ttfmeval.model import build_model


# =============================================================================
# METRICS
# =============================================================================


def get_mean_and_median_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute mean and median MSE, MAE, and MAPE per sample then over samples.

    Args:
        pred: Predictions, shape (N, pred_len) or (N, pred_len, 1).
        gt: Ground truth, same shape as pred.

    Returns:
        Dict with keys mean_mse, mean_mae, mean_mape, median_mse, median_mae, median_mape (scalars).
    """
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
    """Compute MAPE in original scale using per-sample history mean/std.

    Unscales pred and gt then computes |pred - gt| / (|gt| + 1e-8), averaged over
    horizon per sample, then * 100. Shape (N,) returned.

    Args:
        pred: Scaled predictions (N, pred_len) or (N, pred_len, 1).
        gt: Scaled ground truth, same shape as pred.
        history_means: Per-sample history mean, shape (N,) or (N, 1).
        history_stds: Per-sample history std, shape (N,) or (N, 1).

    Returns:
        MAPE per sample in percent, shape (N,).
    """
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
    """Fraction (0-100) of samples where prediction direction matches ground truth at step.

    Direction: sign(pred[step] - last_input) vs sign(gt[step] - last_input). Only
    samples with gt_direction != 0 are counted.

    Args:
        input_arr: Last context value per sample (N,) or (N, 1) or (N, pred_len, 1).
        pred_arr: Predictions (N, pred_len) or (N, pred_len, 1).
        gt_arr: Ground truth, same shape as pred_arr.
        step: 1-based step index (1 = first forecast step).

    Returns:
        Percentage of correct directions, or nan if step out of range, or 0.0 if no valid samples.
    """
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
    """Fraction (0-100) of samples where step-to-step direction matches at given step.

    Direction: sign(pred[step] - pred[step-1]) vs sign(gt[step] - gt[step-1]). Only
    samples with gt_direction != 0 are counted.

    Args:
        pred_arr: Predictions (N, pred_len) or (N, pred_len, 1).
        gt_arr: Ground truth, same shape as pred_arr.
        step: 1-based step index (must be >= 2).

    Returns:
        Percentage of correct step-to-step directions, or nan if step < 2 or out of range.
    """
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
    """Compute scaled metrics and directional accuracy for each model in results.

    Args:
        results: Dict with "input", "gt" (tensors), "predictions" (dict of name -> tensor).
        pred_len: Forecast horizon (for directional accuracy per step). Defaults to 4.

    Returns:
        Dict mapping model_name -> {
            "scaled": get_mean_and_median_metrics output,
            "directional_acc": {step: pct},
            "step_directional_acc": {step: pct},
        }.
    """
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
    """Parse command-line arguments for the evaluation script.

    Returns:
        Namespace with seq_len, pred_len, batch_size, datasets_dir, output_dir,
        device, checkpoint, baseline flags (--eval_<name>), and extra baseline options.
    """
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
        "--datasets_hf",
        type=str,
        default="",
        help="Hugging Face dataset repo id (e.g. bekzatajan/ttfm-sample-datasets). If set, downloads the repo and uses it as datasets_dir.",
    )
    p.add_argument(
        "--datasets_hf_subdir",
        type=str,
        default="",
        help="Subdirectory inside the HF dataset repo to use as datasets_dir. Only used when --datasets_hf is set.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("TTFM_EVAL_OUTPUT_DIR", "./results"),
        help="Output directory for predictions and metrics",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of worker processes for dataset-level parallelism (1=sequential). "
        "With --eval_ttfmlf, all workers share the same vLLM server; start vLLM with "
        "--max-num-seqs >= max_workers so multiple workers can progress in parallel.",
    )
    p.add_argument(
        "--device_ids",
        type=str,
        default="",
        help="Comma-separated GPU IDs for workers (e.g. '0,1,2,7'). If set, workers use cuda:<id> in round-robin.",
    )

    default_checkpoint = os.environ.get("TTFM_CHECKPOINT", "")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help="Hugging Face repo id for TTFM checkpoint (e.g. bekzatajan/ttfm). Or set TTFM_CHECKPOINT env var.",
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
    """Get dataset name from CSV path (basename without .csv).

    Args:
        csv_path: Full path to a CSV file.

    Returns:
        Basename of the file without extension.
    """
    return os.path.splitext(os.path.basename(csv_path))[0]


def get_dataset_output_dir(output_dir: str, dataset_name: str) -> str:
    """Return and create the per-dataset output directory (output_dir/outputs/dataset_name).

    Args:
        output_dir: Root results directory.
        dataset_name: Name of the dataset (e.g. from extract_dataset_name).

    Returns:
        Path to the dataset output directory (created if needed).
    """
    dataset_output_dir = os.path.join(output_dir, "outputs", dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    return dataset_output_dir


def load_per_dataset_results(output_dir: str, dataset_name: str) -> dict | None:
    """Load cached results for a dataset (input.npy, gt.npy, *_pred.npy).

    Args:
        output_dir: Root results directory.
        dataset_name: Name of the dataset.

    Returns:
        Dict with "input", "gt" (tensors), "predictions" (dict of name -> tensor), or None if missing/incomplete.
    """
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


def save_per_dataset_results(output_dir: str, dataset_name: str, results: dict) -> None:
    """Save results to output_dir/outputs/dataset_name (input.npy, gt.npy, *_pred.npy).

    Args:
        output_dir: Root results directory.
        dataset_name: Name of the dataset.
        results: Dict with "input", "gt", "predictions" (same format as baseline eval returns).
    """
    dataset_dir = get_dataset_output_dir(output_dir, dataset_name)
    np.save(os.path.join(dataset_dir, "input.npy"), results["input"].numpy())
    np.save(os.path.join(dataset_dir, "gt.npy"), results["gt"].numpy())
    for model_name, pred_tensor in results["predictions"].items():
        np.save(
            os.path.join(dataset_dir, f"{model_name}_pred.npy"),
            pred_tensor.numpy(),
        )


def get_enabled_baselines(args) -> list:
    """Return list of baseline names for which --eval_<name> was set.

    Args:
        args: Parsed namespace from parse_args().

    Returns:
        List of baseline names (e.g. ["chronos2", "ttfmlf"]).
    """
    enabled = []
    for name in BASELINE_REGISTRY:
        if getattr(args, f"eval_{name}", False):
            enabled.append(name)
    return enabled


def get_expected_prediction_keys(args) -> list:
    """Return all prediction keys that will be produced by enabled baselines.

    Args:
        args: Parsed namespace from parse_args().

    Returns:
        List of unique prediction keys (e.g. ["chronos_univar", "ttfm", "timeseries"]).
    """
    keys = []
    for name in get_enabled_baselines(args):
        keys.extend(BASELINE_REGISTRY[name].prediction_keys)
    return list(set(keys))


def check_cache_status(
    output_dir: str, dataset_name: str, expected_n_samples: int, args
) -> tuple[bool, list]:
    """Check if cached results exist and match expected sample count and keys.

    Args:
        output_dir: Root results directory.
        dataset_name: Name of the dataset.
        expected_n_samples: Expected number of samples (from get_expected_sample_count).
        args: Parsed namespace (used for get_expected_prediction_keys).

    Returns:
        (is_complete, missing_keys): True if all expected *_pred.npy exist and input shape matches; else False and list of missing prediction keys.
    """
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
    """Return the number of test windows for a dataset CSV with current args.

    Args:
        csv_path: Path to the dataset CSV.
        args: Parsed namespace (seq_len, pred_len, etc.).
        LateFusionDataset: The dataset class (to avoid circular import).

    Returns:
        len(dataset) for split="test" with that single CSV.
    """
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample history mean and std for all test windows of a dataset.

    Args:
        csv_path: Path to the dataset CSV.
        args: Parsed namespace (seq_len, pred_len).
        LateFusionDataset: Dataset class.
        late_fusion_collate: Collate function for the dataloader.

    Returns:
        (history_means, history_stds): Each shape (n_test_windows,) for that CSV.
    """
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
    precomputed_results: dict | None = None,
) -> dict | None:
    """Run enabled baselines on a single dataset CSV and merge results.

    Builds a test DataLoader for the CSV, runs each baseline in baselines_to_eval
    (respecting depends_on order), and merges predictions into one result dict.
    If precomputed_results is provided, existing predictions are kept and only
    missing baselines are run.

    Args:
        csv_path: Path to the dataset CSV.
        args: Parsed namespace (batch_size, pred_len, device, etc.).
        LateFusionDataset: Dataset class.
        late_fusion_collate: Collate function.
        models: Dict of loaded models (e.g. {"model": ttfmlf}).
        baselines_to_eval: List of baseline names to run.
        precomputed_results: Optional cached result dict to update. Defaults to None.

    Returns:
        Dict with "input", "gt", "predictions", or None if dataset is empty or evaluation fails.
    """
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

    # Sort so dependencies run before dependents. Example: chronos2_gpt needs
    # gpt_forecast predictions; _run_after_deps(b) is True for dependents (they
    # have a dep in the list), False for baselines with no dep, so sorting by
    # this key puts no-dep first (gpt_forecast) then dependents (chronos2_gpt).
    def _run_after_deps(b):
        d = BASELINE_REGISTRY[b].depends_on
        return d is not None and d in baselines_to_eval

    baselines_to_eval = sorted(baselines_to_eval, key=_run_after_deps)

    univariate_model = "chronos"
    if getattr(args, "univariate_model", None):
        name = str(args.univariate_model).lower()
        if "timesfm" in name:
            univariate_model = "timesfm"
        elif "prophet" in name:
            univariate_model = "prophet"

    for baseline_name in baselines_to_eval:
        config = BASELINE_REGISTRY[baseline_name]
        if all(k in all_results["predictions"] for k in config.prediction_keys):
            continue
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
                dataset_name = extract_dataset_name(csv_path)
                print(
                    f"  Warning: gpt_forecast not available for chronos2_gpt on dataset '{dataset_name}', skipping"
                )
                continue
            # pred_len already passed positionally; omit from kwargs to avoid duplicate
            kwargs_no_pred_len = {k: v for k, v in kwargs.items() if k != "pred_len"}
            result = config.eval_func(
                dataloader,
                args.device,
                args.pred_len,
                precomputed_gpt_forecasts=gpt_forecasts,
                **kwargs_no_pred_len,
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
    """Flatten per-model metrics into a single row for CSV export.

    Args:
        dataset_name: Name of the dataset.
        metrics: Output of compute_all_metrics (model_name -> scaled/directional_acc/step_directional_acc).
        pred_len: Number of steps (for directional columns).

    Returns:
        Dict with dataset_name, and columns like {model}_scaled_mean_mae, {model}_dir_acc_step1, etc.
        If both "timeseries" and "ttfm" exist, adds mae_improvement_pct.
    """
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
    """Return indices of windows whose first date is on or after cutoff_date.

    Args:
        csv_path: Path to CSV with column "t" (parseable dates).
        window_len: Length of each window (seq_len + pred_len).
        cutoff_date: Only windows starting at or after this date. Defaults to "2024-06-01".

    Returns:
        1D array of start indices (0 <= i <= len(df) - window_len).
    """
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
) -> None:
    """Write per-dataset summary CSVs (all samples and June 2024+ filtered).

    Reads cached results from output_dir, computes MAE/MSE/MAPE and TTFM vs timeseries
    improvement, and writes stats_Context_<seq_len>_allsamples.csv and
    stats_Context_<seq_len>_june2024plus.csv.

    Args:
        output_dir: Root results directory.
        args: Parsed namespace (seq_len, datasets_dir).
        LateFusionDataset: Dataset class.
        late_fusion_collate: Collate function (for raw mean/std for MAPE).
    """
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
    """Load TTFM (and TTFM-TimesFM) checkpoints for enabled baselines.

    Args:
        args: Parsed namespace with checkpoint, checkpoint_timesfm, device, text_embedder, etc.

    Returns:
        Dict with keys "model" and/or "model_timesfm" (TTFMLF instances). Empty if neither baseline enabled.

    Raises:
        FileNotFoundError: If --eval_ttfmlf or --eval_ttfmlf_timesfm is set but checkpoint is missing.
    """
    models = {}
    enabled = get_enabled_baselines(args)

    if "ttfmlf" in enabled:
        checkpoint_path = args.checkpoint
        if checkpoint_path and not os.path.isfile(checkpoint_path):
            from ttfmeval.pipeline import _resolve_checkpoint_path

            checkpoint_path = _resolve_checkpoint_path(
                checkpoint_path,
                filename="model.pt",
                token=os.environ.get("HF_TOKEN"),
            )
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "TTFM checkpoint required for --eval_ttfmlf. "
                "Set --checkpoint to a Hugging Face repo id (e.g. bekzatajan/ttfm) "
                "or set TTFM_CHECKPOINT env var."
            )
        print("\nLoading TTFM model...")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
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
# PARALLEL WORKER HELPERS
# =============================================================================

# Module-level state for worker process (one model load per process).
_worker_models = None
_worker_args = None


def _args_to_dict(args) -> dict:
    """Serialize args Namespace to a plain dict for process pool (picklable)."""
    return {k: v for k, v in vars(args).items()}


def _dict_to_args(d: dict):
    """Reconstruct argparse.Namespace from dict (in worker process)."""
    return argparse.Namespace(**d)


def _run_chunk(args_tuple) -> list:
    """Process a chunk of datasets in a worker process. Returns list of (dataset_name, status, detail)."""
    global _worker_models, _worker_args
    chunk, output_dir, args_dict, device_id = args_tuple
    # Lazy-load models once per process with correct device
    if _worker_models is None:
        args = _dict_to_args(args_dict)
        if device_id is not None:
            args.device = f"cuda:{device_id}"
        _worker_args = args
        _worker_models = load_models(args)
    results = []
    for item in chunk:
        csv_path = item["csv_path"]
        dataset_name = item["dataset_name"]
        baselines_to_eval = item["baselines_to_eval"]
        is_partial = item["is_partial"]
        precomputed = (
            load_per_dataset_results(output_dir, dataset_name) if is_partial else None
        )
        try:
            res = evaluate_single_dataset(
                csv_path,
                _worker_args,
                LateFusionDataset,
                late_fusion_collate,
                _worker_models,
                baselines_to_eval,
                precomputed_results=precomputed,
            )
            if res is None:
                results.append((dataset_name, "failed", "no results"))
                continue
            save_per_dataset_results(output_dir, dataset_name, res)
            results.append((dataset_name, "success", res["gt"].shape[0]))
        except Exception as e:
            results.append((dataset_name, "failed", str(e)))
    return results


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry point: parse args, run enabled baselines on datasets, save results and summary CSVs."""
    args = parse_args()

    if getattr(args, "datasets_hf", "") and args.datasets_hf.strip():
        args.datasets_dir = get_datasets_dir_from_hf(
            args.datasets_hf.strip(),
            subdir=getattr(args, "datasets_hf_subdir", "") or None,
            token=os.environ.get("HF_TOKEN"),
        )

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
    if args.max_workers > 1:
        device_ids = (
            [int(x.strip()) for x in args.device_ids.split(",") if x.strip()]
            if args.device_ids
            else None
        )
        print(f"Parallel: max_workers={args.max_workers}, device_ids={device_ids}")
        if "ttfmlf" in enabled_baselines:
            print(
                "  (TTFM uses vLLM for summaries; start vLLM with --max-num-seqs >= "
                f"{args.max_workers} so all workers can run in parallel.)"
            )
    else:
        models = load_models(args)

    print("\n" + "=" * 80)
    print("EVALUATION PLAN")
    print("=" * 80)
    n_skipped = sum(
        1
        for csv_path in dataset_csvs
        if get_expected_sample_count(csv_path, args, LateFusionDataset) <= 0
    )
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
                    "csv_path": csv_path,
                    "dataset": dataset_name,
                    "samples": expected_n_samples,
                    "status": "cached",
                }
            )
        elif missing_keys:
            baselines_to_run = []
            for key in missing_keys:
                baseline = get_baseline_for_prediction_key(key)
                if baseline and baseline in enabled_baselines:
                    baselines_to_run.append(baseline)
            evaluation_plan.append(
                {
                    "csv_path": csv_path,
                    "dataset": dataset_name,
                    "samples": expected_n_samples,
                    "status": "partial",
                    "to_eval": list(dict.fromkeys(baselines_to_run)),
                }
            )
        else:
            evaluation_plan.append(
                {
                    "csv_path": csv_path,
                    "dataset": dataset_name,
                    "samples": expected_n_samples,
                    "status": "full",
                    "to_eval": list(enabled_baselines),
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

    stats = {"successful": 0, "cached": 0, "failed": 0, "skipped": n_skipped}

    # Handle cached datasets (always in main process)
    for plan in evaluation_plan:
        if plan["status"] != "cached":
            continue
        dataset_name = plan["dataset"]
        cached_results = load_per_dataset_results(output_dir, dataset_name)
        if cached_results:
            metrics = compute_all_metrics(cached_results, args.pred_len)
            stats["cached"] += 1
            print(f"\n{dataset_name} (cached): {cached_results['gt'].shape[0]} samples")
            for model_name in sorted(cached_results["predictions"].keys()):
                if model_name in metrics:
                    print(
                        f"  {model_name}: MAE={metrics[model_name]['scaled']['mean_mae']:.4f}"
                    )

    # Build to_run with dependency-ordered baselines
    to_run = []
    for plan in evaluation_plan:
        if plan["status"] == "cached":
            continue
        baselines_to_eval = list(plan["to_eval"])
        for baseline in baselines_to_eval.copy():
            dep = BASELINE_REGISTRY[baseline].depends_on
            if dep and dep not in baselines_to_eval and dep in enabled_baselines:
                baselines_to_eval.insert(0, dep)
        to_run.append(
            {
                "csv_path": plan["csv_path"],
                "dataset_name": plan["dataset"],
                "expected_n_samples": plan["samples"],
                "baselines_to_eval": baselines_to_eval,
                "is_partial": plan["status"] == "partial",
            }
        )

    if args.max_workers == 1:
        # Sequential: run in main process with already-loaded models
        for item in tqdm(to_run, desc="Datasets"):
            csv_path = item["csv_path"]
            dataset_name = item["dataset_name"]
            baselines_to_eval = item["baselines_to_eval"]
            is_partial = item["is_partial"]
            precomputed = (
                load_per_dataset_results(output_dir, dataset_name)
                if is_partial
                else None
            )
            try:
                results = evaluate_single_dataset(
                    csv_path,
                    args,
                    LateFusionDataset,
                    late_fusion_collate,
                    models,
                    baselines_to_eval,
                    precomputed_results=precomputed,
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
    else:
        # Parallel: distribute chunks to workers
        device_ids = (
            [int(x.strip()) for x in args.device_ids.split(",") if x.strip()]
            if args.device_ids
            else None
        )
        args_dict = _args_to_dict(args)
        # Split to_run into max_workers chunks (round-robin for balance)
        n_workers = min(args.max_workers, len(to_run)) if to_run else 0
        chunks = [[] for _ in range(n_workers)]
        for i, item in enumerate(to_run):
            chunks[i % n_workers].append(item)
        tasks = []
        for chunk_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            device_id = (
                device_ids[chunk_index % len(device_ids)] if device_ids else None
            )
            tasks.append((chunk, output_dir, args_dict, device_id))
        if tasks:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_run_chunk, t): t for t in tasks}
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Workers"
                ):
                    try:
                        chunk_results = future.result()
                        for dataset_name, status, detail in chunk_results:
                            if status == "success":
                                stats["successful"] += 1
                                print(f"\n{dataset_name}: {detail} samples")
                            else:
                                stats["failed"] += 1
                                print(f"\n{dataset_name} failed: {detail}")
                    except Exception as e:
                        stats["failed"] += len(futures[future][0])
                        print(f"\nWorker error: {e}")

    generate_overall_stats_csv(output_dir, args, LateFusionDataset, late_fusion_collate)

    print(
        f"\nDone: {stats['successful']} evaluated, {stats['cached']} cached, "
        f"{stats['failed']} failed, {stats['skipped']} skipped."
    )
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
