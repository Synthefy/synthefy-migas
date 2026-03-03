#!/usr/bin/env python3
"""
Generate qualitative forecast plots comparing TTFM and Chronos (or other models).
Finds samples where TTFM has the largest advantage over the baseline model
and creates professional ICML-standard plots showing the forecasts.

Usage:
    python plot_qualitative_forecasts.py --results_dir /path/to/results/context_16 \
                                         --datasets_dir /path/to/datasets \
                                         --output_dir /path/to/output_plots \
                                         --pred_len 4 --top_k 10
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add repo root and src so ttfmeval is importable (synthefy-ttfm repo)
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if _src.exists():
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(_repo_root))


# =============================================================================
# ICML-style plot configuration
# =============================================================================
def setup_icml_style():
    """Configure matplotlib for ICML-standard publication-quality plots."""
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "monospace",
            "font.monospace": ["Roboto Mono", "DejaVu Sans Mono"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # Figure settings
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.format": "png",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Line settings
            "lines.linewidth": 3,
            "lines.markersize": 4,
            # Axes settings
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Legend settings
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.fancybox": False,
            # Use LaTeX for text rendering (if available)
            "text.usetex": False,  # Set to True if LaTeX is installed
            # Tight layout
            "figure.constrained_layout.use": True,
        }
    )


# Define a professional color palette
COLORS = {
    "ground_truth": "#2C3E50",  # Dark blue-gray
    "ttfm": "#27AE60",  # Green (our model)
    "chronos": "#E74C3C",  # Red (baseline)
    "timeseries": "#3498DB",  # Blue
    "input": "#0066CC",  # Darker Blue
    "forecast_region": "#F8F9FA",  # Light gray background for forecast region
}


def _find_dataset_file(datasets_dir, dataset_name: str) -> Path:
    """Resolve dataset file (CSV or Parquet): direct path or recursive search by basename."""
    dd = Path(datasets_dir)
    for ext in (".csv", ".parquet"):
        direct = dd / f"{dataset_name}{ext}"
        if direct.exists():
            return direct
    for ext in (".csv", ".parquet"):
        for p in dd.rglob(f"*{ext}"):
            if p.stem == dataset_name:
                return p
    return dd / f"{dataset_name}.csv"


from ttfmeval.baselines.registry import MODEL_DISPLAY_NAMES


@dataclass
class SampleData:
    """Data class to hold sample information for plotting."""

    sample_idx: int
    dataset_name: str
    local_idx: int
    input_normalized: np.ndarray
    gt_normalized: np.ndarray
    preds_normalized: Dict[str, np.ndarray]
    input_denormalized: np.ndarray
    gt_denormalized: np.ndarray
    preds_denormalized: Dict[str, np.ndarray]
    history_mean: float
    history_std: float
    mae_ttfm: float
    mae_baseline: float
    mae_advantage: float
    # New fields for dates and text
    dates: Optional[List[pd.Timestamp]] = None
    text_annotations: Optional[List[str]] = None
    context_summary: Optional[str] = None


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute MAE for each sample."""
    return np.mean(np.abs(pred - gt), axis=-1)


def load_sample_dates_and_text(
    data_path: Path,
    local_idx: int,
    context_len: int,
    pred_len: int,
    val_length: int = 1000,
) -> Tuple[List[pd.Timestamp], List[str]]:
    """
    Load dates and text annotations for a specific sample from the data file.

    Args:
        data_path: Path to the dataset file (CSV or Parquet)
        local_idx: Local sample index within this dataset (0-based)
        context_len: Context length
        pred_len: Prediction length
        val_length: Validation set length (samples taken from end)

    Returns:
        Tuple of (dates, text_annotations) for the full window (context + pred)
    """
    ext = (
        data_path.suffix.lower()
        if hasattr(data_path, "suffix")
        else os.path.splitext(str(data_path))[1].lower()
    )
    df = pd.read_parquet(data_path) if ext == ".parquet" else pd.read_csv(data_path)
    df["t"] = pd.to_datetime(df["t"])

    total_len = len(df)
    window_size = context_len + pred_len

    # Validation set uses the last val_length samples with stride=1
    # Max possible samples = total_len - window_size + 1
    max_samples = total_len - window_size + 1
    actual_val_length = min(val_length, max_samples)

    # The validation set starts at: total_len - actual_val_length - window_size + 1
    val_start_idx = total_len - actual_val_length - window_size + 1

    # Ensure val_start_idx is not negative
    val_start_idx = max(0, val_start_idx)

    # Sample local_idx corresponds to starting at: val_start_idx + local_idx
    sample_start = val_start_idx + local_idx
    sample_end = sample_start + window_size

    # Bounds check
    if sample_start < 0 or sample_end > total_len:
        raise ValueError(
            f"Sample index {local_idx} out of bounds for dataset with {total_len} rows "
            f"(window_size={window_size}, sample_start={sample_start}, sample_end={sample_end})"
        )

    # Extract dates and text for this window
    dates = df["t"].iloc[sample_start:sample_end].tolist()

    # Handle text column - might be 'text' or might not exist
    if "text" in df.columns:
        text_annotations = df["text"].iloc[sample_start:sample_end].fillna("").tolist()
    else:
        text_annotations = [""] * window_size

    return dates, text_annotations


def generate_context_summaries(
    samples: List["SampleData"],
    datasets_dir: Path,
    context_len: int,
    pred_len: int,
    llm_base_url: str = "http://localhost:8004/v1",
    llm_model: str = "openai/gpt-oss-120b",
) -> List["SampleData"]:
    """
    Generate context summaries for a list of samples using the LLM.

    Args:
        samples: List of SampleData objects (will be modified in place)
        datasets_dir: Directory containing dataset CSVs
        context_len: Context length
        pred_len: Prediction length
        llm_base_url: vLLM server URL
        llm_model: Model name

    Returns:
        The same list of samples with context_summary filled in
    """
    try:
        from ttfmeval.model.util import ContextSummarizer
    except ImportError:
        try:
            from src.context_summary import ContextSummarizer
        except ImportError:
            print(
                "  Skipping context summaries (ttfmeval.model.util.ContextSummarizer not available; install synthefy-ttfm or add ContextSummarizer for LLM summaries)."
            )
            return samples

    print(f"\nGenerating context summaries for {len(samples)} samples...")

    # Initialize summarizer
    summarizer = ContextSummarizer(
        base_url=llm_base_url,
        model_name=llm_model,
        max_concurrent=32,
    )

    # Load dates and text for each sample
    loaded_count = 0
    for i, sample in enumerate(samples):
        data_path = _find_dataset_file(datasets_dir, sample.dataset_name)
        if data_path.exists():
            try:
                dates, text_annotations = load_sample_dates_and_text(
                    data_path, sample.local_idx, context_len, pred_len
                )
                sample.dates = dates
                sample.text_annotations = text_annotations
                loaded_count += 1
            except Exception as e:
                print(
                    f"  Warning: Could not load dates/text for {sample.dataset_name} sample {sample.local_idx}: {e}"
                )
                sample.dates = None
                sample.text_annotations = None
        else:
            print(f"  Warning: Data file not found: {data_path}")
            sample.dates = None
            sample.text_annotations = None

    print(f"  Loaded dates/text for {loaded_count}/{len(samples)} samples")

    # Prepare batches for summarization
    text_batch = []
    values_batch = []
    valid_indices = []
    skipped_no_text = 0
    skipped_no_annotations = 0

    for i, sample in enumerate(samples):
        if sample.text_annotations is None:
            skipped_no_annotations += 1
            continue

        # Only use context portion for summary
        context_text = sample.text_annotations[:context_len]
        context_values = sample.input_denormalized.tolist()

        # Check if there's any text (but don't skip if all empty - still generate summary with values only)
        has_text = any(t and isinstance(t, str) and t.strip() for t in context_text)

        if has_text:
            text_batch.append(context_text)
            values_batch.append(context_values)
            valid_indices.append(i)
        else:
            skipped_no_text += 1
            # Still generate a simple summary even without text
            sample.context_summary = (
                "No text annotations available for this time window."
            )

    if skipped_no_annotations > 0:
        print(
            f"  Skipped {skipped_no_annotations} samples (could not load annotations)"
        )
    if skipped_no_text > 0:
        print(f"  Skipped {skipped_no_text} samples (no text in context window)")

    if text_batch:
        print(f"  Summarizing {len(text_batch)} samples with text...")
        try:
            summaries = summarizer.summarize_batch(text_batch, values_batch)

            for idx, summary in zip(valid_indices, summaries):
                samples[idx].context_summary = summary

            print(f"  Generated {len(summaries)} summaries")
        except Exception as e:
            print(f"  Warning: Could not generate summaries: {e}")
            import traceback

            traceback.print_exc()

    return samples


def load_predictions_from_outputs(
    results_dir: Path, per_dataset_csv: Path
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load all prediction files from the new per-dataset outputs structure.

    The new structure is:
        results_dir/
          outputs/
            {dataset_name}/
              input.npy
              gt.npy
              {model}_pred.npy
          per_dataset_metrics.csv

    Returns:
        Tuple of (predictions dict, gt array, input array) with all datasets concatenated
    """
    outputs_dir = results_dir / "outputs"

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    # Read per-dataset CSV to get dataset order
    df = pd.read_csv(per_dataset_csv)

    # Known model names to look for
    model_names = [
        "ttfm",
        "timeseries",
        "chronos_univar",
        "chronos_multivar",
        "chronos_emb",
        "chronos_gpt_cov",
        "chronos_gpt_dir_cov",
        "gpt_forecast",
        "timesfm_univar",
        "tabpfn_ts",
        "prophet",
        "naive",
        "ttfm_timesfm",
        "migas",
    ]

    predictions = {name: [] for name in model_names}
    all_gt = []
    all_input = []

    print(f"\nLoading predictions from {len(df)} datasets...")

    for idx, row in df.iterrows():
        dataset_name = row["dataset_name"]
        dataset_dir = outputs_dir / dataset_name

        if not dataset_dir.exists():
            print(f"  Warning: Dataset directory not found: {dataset_dir}")
            continue

        # Load gt and input
        gt_path = dataset_dir / "gt.npy"
        input_path = dataset_dir / "input.npy"

        if not gt_path.exists() or not input_path.exists():
            print(f"  Warning: Missing gt.npy or input.npy for {dataset_name}")
            continue

        gt = np.load(gt_path)
        inp = np.load(input_path)
        all_gt.append(gt)
        all_input.append(inp)

        # Load predictions for each model
        for model_name in model_names:
            pred_path = dataset_dir / f"{model_name}_pred.npy"
            if pred_path.exists():
                predictions[model_name].append(np.load(pred_path))

    # Concatenate all data
    gt_concat = np.concatenate(all_gt, axis=0)
    input_concat = np.concatenate(all_input, axis=0)

    # Concatenate predictions and remove empty ones
    predictions_concat = {}
    for model_name, pred_list in predictions.items():
        if pred_list:
            predictions_concat[model_name] = np.concatenate(pred_list, axis=0)
            print(
                f"  Loaded {model_name}: shape {predictions_concat[model_name].shape}"
            )

    print(f"  Ground truth shape: {gt_concat.shape}")
    print(f"  Input shape: {input_concat.shape}")

    return predictions_concat, gt_concat, input_concat


def load_predictions(results_dir: Path) -> Dict[str, np.ndarray]:
    """Load all prediction files from the results directory (legacy format)."""
    predictions = {}

    # Map file names to model names
    pred_files = {
        "ttfm": "ttfm_pred.npy",
        "timeseries": "timeseries_pred.npy",
        "chronos_univar": "chronos_univar_pred.npy",
        "chronos_multivar": "chronos_multivar_pred.npy",
        "chronos_emb": "chronos_emb_pred.npy",
        "chronos_gpt_cov": "chronos_gpt_cov_pred.npy",
        "chronos_gpt_dir_cov": "chronos_gpt_dir_cov_pred.npy",
        "gpt_forecast": "gpt_forecast_pred.npy",
        "timesfm_univar": "timesfm_univar_pred.npy",
        "tabpfn_ts": "tabpfn_ts_pred.npy",
    }

    for model_name, filename in pred_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            predictions[model_name] = np.load(filepath)
            print(f"  Loaded {model_name}: shape {predictions[model_name].shape}")

    return predictions


def compute_raw_mean_std(
    per_dataset_csv: Path, datasets_dir: Path, context_len: int, pred_len: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute raw (unscaled) mean and std for each sample using the dataset loader.
    Also returns the dataset name for each sample.

    Returns:
        Tuple of (history_means, history_stds, dataset_names_per_sample) arrays
    """
    from ttfmeval.dataset import LateFusionDataset, collate_fn as late_fusion_collate
    from torch.utils.data import DataLoader

    df = pd.read_csv(per_dataset_csv)
    sample_count_col = (
        "n_eval_samples" if "n_eval_samples" in df.columns else "n_samples"
    )
    history_means = []
    history_stds = []
    dataset_names_per_sample = []

    print(f"\nLoading normalization parameters from {len(df)} datasets...")

    for idx, row in df.iterrows():
        dataset_name = row["dataset_name"]
        n_samples_expected = int(row[sample_count_col])
        data_path = _find_dataset_file(datasets_dir, dataset_name)

        if not data_path.exists():
            print(f"  Warning: {data_path} not found, using default normalization")
            history_means.extend([0.0] * n_samples_expected)
            history_stds.extend([1.0] * n_samples_expected)
            dataset_names_per_sample.extend([dataset_name] * n_samples_expected)
            continue

        # Use dataset loader to get history_means and history_stds
        try:
            dataset = LateFusionDataset(
                context_len + pred_len,
                pred_len,
                [str(data_path)],
                split="test",
                val_length=1000,
                stride=1,
            )

            dataloader = DataLoader(
                dataset, batch_size=32, shuffle=False, collate_fn=late_fusion_collate
            )

            dataset_means = []
            dataset_stds = []
            for batch_dict in dataloader:
                batch_means = batch_dict["history_means"]
                batch_stds = batch_dict["history_stds"]
                dataset_means.extend(batch_means)
                dataset_stds.extend(batch_stds)

            n_samples_loaded = len(dataset_means)
            if n_samples_loaded != n_samples_expected:
                print(
                    f"  Warning: Sample count mismatch for {dataset_name}: "
                    f"expected {n_samples_expected}, got {n_samples_loaded}"
                )
                # Pad or truncate to expected count
                if n_samples_loaded < n_samples_expected:
                    dataset_means.extend(
                        [dataset_means[-1]] * (n_samples_expected - n_samples_loaded)
                    )
                    dataset_stds.extend(
                        [dataset_stds[-1]] * (n_samples_expected - n_samples_loaded)
                    )
                else:
                    dataset_means = dataset_means[:n_samples_expected]
                    dataset_stds = dataset_stds[:n_samples_expected]

            history_means.extend(dataset_means)
            history_stds.extend(dataset_stds)
            dataset_names_per_sample.extend([dataset_name] * n_samples_expected)

        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            history_means.extend([0.0] * n_samples_expected)
            history_stds.extend([1.0] * n_samples_expected)
            dataset_names_per_sample.extend([dataset_name] * n_samples_expected)

    return np.array(history_means), np.array(history_stds), dataset_names_per_sample


def find_absolute_best_samples(
    predictions: Dict[str, np.ndarray],
    gt: np.ndarray,
    input_data: np.ndarray,
    history_means: np.ndarray,
    history_stds: np.ndarray,
    dataset_names: List[str],
    per_dataset_csv: Path,
    model_name: str = "ttfm",
    top_k: int = 10,
    per_dataset: bool = True,
) -> List[SampleData]:
    """
    Find samples with the lowest absolute MAE for a given model.

    Args:
        predictions: Dict of model predictions
        gt: Ground truth array
        input_data: Input array
        history_means: Normalization means
        history_stds: Normalization stds
        dataset_names: Dataset name for each sample
        per_dataset_csv: Path to per-dataset metrics CSV
        model_name: Name of model to find best samples for
        top_k: Number of top samples to return (per dataset if per_dataset=True)
        per_dataset: If True, return top_k samples per dataset

    Returns:
        List of SampleData objects for the samples with lowest MAE
    """
    if model_name not in predictions:
        raise ValueError(f"Model '{model_name}' not found in predictions")

    model_pred = predictions[model_name]

    # Compute MAE per sample
    mae_model = compute_mae(model_pred, gt)

    print(
        f"\n  {model_name} MAE stats: min={mae_model.min():.4f}, "
        f"median={np.median(mae_model):.4f}, max={mae_model.max():.4f}"
    )

    best_samples = []

    if per_dataset:
        # Get top_k samples per dataset (lowest MAE)
        df = pd.read_csv(per_dataset_csv)
        sample_count_col = (
            "n_eval_samples" if "n_eval_samples" in df.columns else "n_samples"
        )
        current_idx = 0

        for _, row in df.iterrows():
            dataset_name = row["dataset_name"]
            n_samples = int(row[sample_count_col])

            # Get indices for this dataset
            dataset_slice = slice(current_idx, current_idx + n_samples)
            dataset_mae = mae_model[dataset_slice]

            # Sort by MAE (ascending - lowest first) and take top_k
            sorted_local_indices = np.argsort(dataset_mae)[:top_k]

            print(
                f"  {dataset_name}: Selecting {len(sorted_local_indices)} best samples "
                f"(MAE range: {dataset_mae[sorted_local_indices[0]]:.4f} - "
                f"{dataset_mae[sorted_local_indices[-1]]:.4f})"
            )

            for local_idx in sorted_local_indices:
                global_idx = current_idx + local_idx

                # Get normalized data
                input_norm = input_data[global_idx]
                gt_norm = gt[global_idx]
                preds_norm = {k: v[global_idx] for k, v in predictions.items()}

                # Denormalize
                mean = history_means[global_idx]
                std = history_stds[global_idx]

                input_denorm = input_norm * std + mean
                gt_denorm = gt_norm * std + mean
                preds_denorm = {k: v * std + mean for k, v in preds_norm.items()}

                sample = SampleData(
                    sample_idx=global_idx,
                    dataset_name=dataset_name,
                    local_idx=local_idx,
                    input_normalized=input_norm,
                    gt_normalized=gt_norm,
                    preds_normalized=preds_norm,
                    input_denormalized=input_denorm,
                    gt_denormalized=gt_denorm,
                    preds_denormalized=preds_denorm,
                    history_mean=mean,
                    history_std=std,
                    mae_ttfm=mae_model[global_idx],
                    mae_baseline=mae_model[
                        global_idx
                    ],  # Same as ttfm for absolute best
                    mae_advantage=0.0,  # No comparison
                )
                best_samples.append(sample)

            current_idx += n_samples
    else:
        # Global top_k (lowest MAE)
        sorted_indices = np.argsort(mae_model)[:top_k]

        for global_idx in sorted_indices:
            # Get normalized data
            input_norm = input_data[global_idx]
            gt_norm = gt[global_idx]
            preds_norm = {k: v[global_idx] for k, v in predictions.items()}

            # Denormalize
            mean = history_means[global_idx]
            std = history_stds[global_idx]

            input_denorm = input_norm * std + mean
            gt_denorm = gt_norm * std + mean
            preds_denorm = {k: v * std + mean for k, v in preds_norm.items()}

            sample = SampleData(
                sample_idx=global_idx,
                dataset_name=dataset_names[global_idx],
                local_idx=global_idx,
                input_normalized=input_norm,
                gt_normalized=gt_norm,
                preds_normalized=preds_norm,
                input_denormalized=input_denorm,
                gt_denormalized=gt_denorm,
                preds_denormalized=preds_denorm,
                history_mean=mean,
                history_std=std,
                mae_ttfm=mae_model[global_idx],
                mae_baseline=mae_model[global_idx],
                mae_advantage=0.0,
            )
            best_samples.append(sample)

    return best_samples


def find_best_samples(
    predictions: Dict[str, np.ndarray],
    gt: np.ndarray,
    input_data: np.ndarray,
    history_means: np.ndarray,
    history_stds: np.ndarray,
    dataset_names: List[str],
    per_dataset_csv: Path,
    ttfm_model: str = "ttfm",
    baseline_model: str = "chronos_univar",
    top_k: int = 10,
    per_dataset: bool = True,
) -> List[SampleData]:
    """
    Find samples where TTFM has the largest advantage over the baseline model.

    Args:
        predictions: Dict of model predictions
        gt: Ground truth array
        input_data: Input array
        history_means: Normalization means
        history_stds: Normalization stds
        dataset_names: Dataset name for each sample
        per_dataset_csv: Path to per-dataset metrics CSV
        ttfm_model: Name of TTFM model
        baseline_model: Name of baseline model to compare against
        top_k: Number of top samples to return (per dataset if per_dataset=True)
        per_dataset: If True, return top_k samples per dataset

    Returns:
        List of SampleData objects for the best samples
    """
    if ttfm_model not in predictions:
        raise ValueError(f"TTFM model '{ttfm_model}' not found in predictions")
    if baseline_model not in predictions:
        raise ValueError(f"Baseline model '{baseline_model}' not found in predictions")

    ttfm_pred = predictions[ttfm_model]
    baseline_pred = predictions[baseline_model]

    # Compute MAE per sample
    mae_ttfm = compute_mae(ttfm_pred, gt)
    mae_baseline = compute_mae(baseline_pred, gt)

    # Compute advantage (positive = TTFM better)
    mae_advantage = mae_baseline - mae_ttfm

    # Filter to samples where TTFM is better
    ttfm_better_mask = mae_advantage > 0

    print(
        f"\n  TTFM wins on {ttfm_better_mask.sum()}/{len(mae_advantage)} samples "
        f"({100 * ttfm_better_mask.sum() / len(mae_advantage):.1f}%)"
    )

    best_samples = []

    if per_dataset:
        # Get top_k samples per dataset
        df = pd.read_csv(per_dataset_csv)
        sample_count_col = (
            "n_eval_samples" if "n_eval_samples" in df.columns else "n_samples"
        )
        current_idx = 0

        for _, row in df.iterrows():
            dataset_name = row["dataset_name"]
            n_samples = int(row[sample_count_col])

            # Get indices for this dataset
            dataset_slice = slice(current_idx, current_idx + n_samples)
            dataset_advantage = mae_advantage[dataset_slice]
            dataset_ttfm_better = ttfm_better_mask[dataset_slice]

            # Find indices where TTFM is better
            better_local_indices = np.where(dataset_ttfm_better)[0]

            if len(better_local_indices) == 0:
                print(f"  {dataset_name}: No samples where TTFM is better")
                current_idx += n_samples
                continue

            # Sort by advantage and take top_k
            sorted_indices = better_local_indices[
                np.argsort(dataset_advantage[better_local_indices])[::-1]
            ][:top_k]

            print(
                f"  {dataset_name}: Found {len(better_local_indices)} samples, "
                f"selecting top {len(sorted_indices)}"
            )

            for local_idx in sorted_indices:
                global_idx = current_idx + local_idx

                # Get normalized data
                input_norm = input_data[global_idx]
                gt_norm = gt[global_idx]
                preds_norm = {k: v[global_idx] for k, v in predictions.items()}

                # Denormalize
                mean = history_means[global_idx]
                std = history_stds[global_idx]

                input_denorm = input_norm * std + mean
                gt_denorm = gt_norm * std + mean
                preds_denorm = {k: v * std + mean for k, v in preds_norm.items()}

                sample = SampleData(
                    sample_idx=global_idx,
                    dataset_name=dataset_name,
                    local_idx=local_idx,
                    input_normalized=input_norm,
                    gt_normalized=gt_norm,
                    preds_normalized=preds_norm,
                    input_denormalized=input_denorm,
                    gt_denormalized=gt_denorm,
                    preds_denormalized=preds_denorm,
                    history_mean=mean,
                    history_std=std,
                    mae_ttfm=mae_ttfm[global_idx],
                    mae_baseline=mae_baseline[global_idx],
                    mae_advantage=mae_advantage[global_idx],
                )
                best_samples.append(sample)

            current_idx += n_samples
    else:
        # Global top_k
        better_indices = np.where(ttfm_better_mask)[0]
        sorted_indices = better_indices[
            np.argsort(mae_advantage[better_indices])[::-1]
        ][:top_k]

        for global_idx in sorted_indices:
            # Get normalized data
            input_norm = input_data[global_idx]
            gt_norm = gt[global_idx]
            preds_norm = {k: v[global_idx] for k, v in predictions.items()}

            # Denormalize
            mean = history_means[global_idx]
            std = history_stds[global_idx]

            input_denorm = input_norm * std + mean
            gt_denorm = gt_norm * std + mean
            preds_denorm = {k: v * std + mean for k, v in preds_norm.items()}

            sample = SampleData(
                sample_idx=global_idx,
                dataset_name=dataset_names[global_idx],
                local_idx=global_idx,  # Not meaningful for global selection
                input_normalized=input_norm,
                gt_normalized=gt_norm,
                preds_normalized=preds_norm,
                input_denormalized=input_denorm,
                gt_denormalized=gt_denorm,
                preds_denormalized=preds_denorm,
                history_mean=mean,
                history_std=std,
                mae_ttfm=mae_ttfm[global_idx],
                mae_baseline=mae_baseline[global_idx],
                mae_advantage=mae_advantage[global_idx],
            )
            best_samples.append(sample)

    return best_samples


def _create_forecast_plot(
    sample: SampleData,
    models_to_plot: List[str],
    show_normalized: bool = False,
    figsize: Tuple[float, float] = (7.5, 4.5),
    show_legend: bool = True,
    show_title: bool = True,
    show_summary: bool = False,
    rotate_xticks: bool = True,
    show_mae: bool = True,
) -> plt.Figure:
    """
    Create a forecast plot figure (helper function).

    Args:
        sample: SampleData object
        models_to_plot: List of model names to include
        show_normalized: If True, plot normalized values
        figsize: Figure size
        show_legend: If True, show legend
        show_title: If True, show title
        show_summary: If True, add context summary below plot
        rotate_xticks: If True, rotate date x-ticks 45 degrees
        show_mae: If True, show MAE annotation box

    Returns:
        matplotlib Figure object
    """
    import matplotlib.dates as mdates
    import textwrap

    setup_icml_style()

    # Adjust figure height if showing summary
    if show_summary and sample.context_summary:
        fig_height = figsize[1] + 1.5
    else:
        fig_height = figsize[1]

    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))

    # Choose data
    if show_normalized:
        input_data = sample.input_normalized
        gt_data = sample.gt_normalized
        preds_data = sample.preds_normalized
        ylabel = "Normalized Value"
    else:
        input_data = sample.input_denormalized
        gt_data = sample.gt_denormalized
        preds_data = sample.preds_denormalized
        ylabel = "Value"

    context_len = len(input_data)
    pred_len = len(gt_data)

    # Get last input value for continuity
    last_input = input_data[-1]
    gt_extended = np.concatenate([[last_input], gt_data])

    # Model colors
    model_colors = {
        "ttfm": COLORS["ttfm"],
        "chronos_univar": COLORS["chronos"],
        "timeseries": COLORS["timeseries"],
        "timesfm_univar": "#9B59B6",
        "gpt_forecast": "#F39C12",
        "migas": "#E67E22",
    }

    # Check if we have dates
    use_dates = sample.dates is not None and len(sample.dates) == context_len + pred_len

    if use_dates:
        dates_context = sample.dates[:context_len]
        dates_pred = sample.dates[context_len:]
        dates_pred_extended = [dates_context[-1]] + list(dates_pred)

        # Shaded forecast region
        ax.axvspan(dates_context[-1], dates_pred[-1], alpha=0.15, color="gray")
        ax.axvline(
            x=dates_context[-1], color="gray", linestyle="--", linewidth=1, alpha=0.7
        )

        # Plot input context
        ax.plot(
            dates_context,
            input_data,
            color=COLORS["input"],
            linewidth=3,
            label="Historical",
            zorder=3,
        )

        # Plot ground truth
        ax.plot(
            dates_pred_extended,
            gt_extended,
            color=COLORS["ground_truth"],
            linewidth=3,
            label="Ground Truth",
            zorder=4,
        )

        # Plot predictions
        for model_name in models_to_plot:
            if model_name not in preds_data:
                continue
            pred = preds_data[model_name]
            pred_extended = np.concatenate([[last_input], pred])
            color = model_colors.get(model_name, "#95A5A6")
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            ax.plot(
                dates_pred_extended,
                pred_extended,
                color=color,
                linewidth=3,
                label=display_name,
                zorder=5 if model_name == "ttfm" else 4,
                alpha=0.9,
            )

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if rotate_xticks:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Date")
    else:
        # Use step indices
        t_input = np.arange(context_len)
        t_pred_extended = np.arange(context_len - 1, context_len + pred_len)

        # Shaded forecast region
        ax.axvspan(
            context_len - 0.5, context_len + pred_len - 0.5, alpha=0.15, color="gray"
        )
        ax.axvline(
            x=context_len - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7
        )

        # Plot input context
        ax.plot(
            t_input,
            input_data,
            color=COLORS["input"],
            linewidth=3,
            label="Historical",
            zorder=3,
        )

        # Plot ground truth
        ax.plot(
            t_pred_extended,
            gt_extended,
            color=COLORS["ground_truth"],
            linewidth=3,
            label="Ground Truth",
            zorder=4,
        )

        # Plot predictions
        for model_name in models_to_plot:
            if model_name not in preds_data:
                continue
            pred = preds_data[model_name]
            pred_extended = np.concatenate([[last_input], pred])
            color = model_colors.get(model_name, "#95A5A6")
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            ax.plot(
                t_pred_extended,
                pred_extended,
                color=color,
                linewidth=3,
                label=display_name,
                zorder=5 if model_name == "ttfm" else 4,
                alpha=0.9,
            )

        ax.set_xlabel("Time Step")
        ax.set_xlim(-0.5, context_len + pred_len - 0.5)
        ax.set_xticks(
            np.arange(0, context_len + pred_len, max(1, (context_len + pred_len) // 10))
        )

    # Labels
    ax.set_ylabel(ylabel)

    # Title
    if show_title:
        display_dataset = (
            sample.dataset_name.replace("_", " ").replace("with text", "").strip()
        )
        if len(display_dataset) > 40:
            display_dataset = display_dataset[:37] + "..."
        ax.set_title(display_dataset, fontweight="bold", fontsize=11)

    # MAE annotation
    if show_mae:
        mae_ttfm_str = (
            f"{sample.mae_ttfm:.4f}"
            if show_normalized
            else f"{sample.mae_ttfm * sample.history_std:.4f}"
        )
        mae_base_str = (
            f"{sample.mae_baseline:.4f}"
            if show_normalized
            else f"{sample.mae_baseline * sample.history_std:.4f}"
        )
        mae_text = f"MAE: TTFM={mae_ttfm_str}, Chronos2={mae_base_str}"
        ax.text(
            0.02,
            0.98,
            mae_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9
            ),
        )

    # Legend
    if show_legend:
        ax.legend(
            loc="upper left",
            framealpha=0.95,
            fontsize=8,
            ncol=2 if len(models_to_plot) > 3 else 1,
        )

    # Grid
    ax.grid(True, which="major", linestyle="-", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    # Context summary
    if show_summary and sample.context_summary:
        # Escape special characters that matplotlib interprets as LaTeX
        safe_summary = sample.context_summary
        for char in ["$", "%", "_", "^", "&", "#", "{", "}", "~"]:
            safe_summary = safe_summary.replace(char, "\\" + char)
        wrapped_summary = textwrap.fill(safe_summary, width=100)
        fig.text(
            0.5,
            -0.02,
            wrapped_summary,
            ha="center",
            va="top",
            fontsize=8,
            wrap=True,
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightyellow",
                edgecolor="gray",
                alpha=0.9,
            ),
        )

    plt.tight_layout()
    return fig


def plot_single_forecast(
    sample: SampleData,
    models_to_plot: List[str],
    output_path: Path,
    show_normalized: bool = False,
    figsize: Tuple[float, float] = (7.5, 4.5),
    show_summary: bool = True,
):
    """
    Create a professional ICML-style plot for a single forecast.

    When show_summary is True and sample has context_summary:
    - Saves plain version (no legend, no title, horizontal x-ticks) as sample{idx}.png
    - Saves version with text summary as sample{idx}_txt.png

    Args:
        sample: SampleData object containing all sample information
        models_to_plot: List of model names to include in the plot
        output_path: Path to save the plot (base path, will add _txt suffix for text version)
        show_normalized: If True, plot normalized values; otherwise denormalized
        figsize: Figure size in inches
        show_summary: If True, also save a version with context summary text
    """
    has_summary = show_summary and sample.context_summary

    if has_summary:
        # Save plain version (no legend, no title, no rotated ticks, no MAE) - wider rectangular aspect
        plain_figsize = (
            figsize[0] * 1.3,
            figsize[1] * 0.9,
        )  # Wider and shorter for rectangular look
        fig_plain = _create_forecast_plot(
            sample=sample,
            models_to_plot=models_to_plot,
            show_normalized=show_normalized,
            figsize=plain_figsize,
            show_legend=False,
            show_title=False,
            show_summary=False,
            rotate_xticks=False,
            show_mae=False,
        )
        fig_plain.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig_plain)

        # Save version with text summary
        output_path_txt = output_path.parent / f"{output_path.stem}_txt.png"
        fig_txt = _create_forecast_plot(
            sample=sample,
            models_to_plot=models_to_plot,
            show_normalized=show_normalized,
            figsize=figsize,
            show_legend=True,
            show_title=True,
            show_summary=True,
            rotate_xticks=True,
        )
        fig_txt.savefig(output_path_txt, dpi=300, bbox_inches="tight")
        plt.close(fig_txt)
    else:
        # Save single version with legend and title
        fig = _create_forecast_plot(
            sample=sample,
            models_to_plot=models_to_plot,
            show_normalized=show_normalized,
            figsize=figsize,
            show_legend=True,
            show_title=True,
            show_summary=False,
            rotate_xticks=True,
        )
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_multi_sample_comparison(
    samples: List[SampleData],
    models_to_plot: List[str],
    output_path: Path,
    n_cols: int = 2,
    figsize_per_subplot: Tuple[float, float] = (5.5, 2.8),
):
    """
    Create a grid of forecast plots for multiple samples.

    Args:
        samples: List of SampleData objects
        models_to_plot: List of model names to include
        output_path: Path to save the plot
        n_cols: Number of columns in the grid
        figsize_per_subplot: Size of each subplot (width, height) - wide rectangular format
    """
    setup_icml_style()

    n_samples = len(samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    # Scale figure size based on number of samples - each subplot is wide/rectangular
    figsize = (figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    model_colors = {
        "ttfm": COLORS["ttfm"],
        "chronos_univar": COLORS["chronos"],
        "timeseries": COLORS["timeseries"],
        "timesfm_univar": "#9B59B6",
        "gpt_forecast": "#F39C12",
        "migas": "#E67E22",
    }

    model_markers = {
        "ttfm": "^",
        "chronos_univar": "v",
        "timeseries": "D",
        "timesfm_univar": "p",
        "gpt_forecast": "*",
        "migas": "h",
    }

    for idx, sample in enumerate(samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        input_data = sample.input_denormalized
        gt_data = sample.gt_denormalized
        preds_data = sample.preds_denormalized

        context_len = len(input_data)
        pred_len = len(gt_data)
        last_input = input_data[-1]

        t_input = np.arange(context_len)
        # Extended time for forecast (includes last context point)
        t_pred_extended = np.arange(context_len - 1, context_len + pred_len)

        # Shaded forecast region
        ax.axvspan(
            context_len - 0.5, context_len + pred_len - 0.5, alpha=0.15, color="gray"
        )
        ax.axvline(
            x=context_len - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7
        )

        # Plot input
        ax.plot(
            t_input,
            input_data,
            color=COLORS["input"],
            linewidth=3,
            # marker='o', markersize=3,
            label="Historical",
        )

        # Plot ground truth (extended to include last context point)
        gt_extended = np.concatenate([[last_input], gt_data])
        ax.plot(
            t_pred_extended,
            gt_extended,
            color=COLORS["ground_truth"],
            linewidth=3,
            # marker='s', markersize=4,
            label="Ground Truth",
        )

        # Plot predictions (extended to include last context point)
        for model_name in models_to_plot:
            if model_name not in preds_data:
                continue
            pred = preds_data[model_name]
            pred_extended = np.concatenate([[last_input], pred])
            color = model_colors.get(model_name, "#95A5A6")
            marker = model_markers.get(model_name, "o")
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

            ax.plot(
                t_pred_extended,
                pred_extended,
                color=color,
                linewidth=3,
                # marker=marker, markersize=4,
                label=display_name,
            )

        # Title - clean up dataset name and add sample number
        display_dataset = (
            sample.dataset_name.replace("_", " ").replace("with text", "").strip()
        )
        ax.set_title(
            f"{display_dataset} (Sample {sample.sample_idx})",
            fontsize=10,
            fontweight="bold",
        )

        # Axis labels (only on edges)
        if row == n_rows - 1:
            ax.set_xlabel("Time Step", fontsize=10)
        if col == 0:
            ax.set_ylabel("Value", fontsize=10)

        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Shared legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(models_to_plot) + 2,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
        framealpha=0.95,
    )

    plt.tight_layout()
    # Adjust spacing: more space between subplots and room for legend at bottom
    plt.subplots_adjust(bottom=0.08, hspace=0.35, wspace=0.25)

    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative forecast comparison plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic usage - find samples where TTFM beats Chronos
  python plot_qualitative_forecasts.py \\
      --results_dir /path/to/results/context_16 \\
      --datasets_dir /path/to/datasets \\
      --pred_len 4 --top_k 10

  # Find samples with lowest absolute MAE for TTFM (best predictions)
  python plot_qualitative_forecasts.py \\
      --results_dir /path/to/results/context_16 \\
      --datasets_dir /path/to/datasets \\
      --absolute_best --better_model ttfm

  # Compare TimesFM vs TTFM (samples where TimesFM is better)
  python plot_qualitative_forecasts.py \\
      --results_dir /path/to/results/context_16 \\
      --datasets_dir /path/to/datasets \\
      --better_model timesfm_univar --worse_model ttfm \\
      --models_to_plot ttfm,timesfm_univar,chronos_univar

  # Create grid plots showing multiple samples per figure
  python plot_qualitative_forecasts.py \\
      --results_dir /path/to/results/context_16 \\
      --datasets_dir /path/to/datasets \\
      --create_grid --grid_samples 6
""",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing prediction .npy files",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        required=True,
        help="Directory containing original dataset CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results_dir/qualitative_plots)",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="Context length (auto-detected from directory name if not provided)",
    )
    parser.add_argument("--pred_len", type=int, default=4, help="Prediction length")
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top samples per dataset"
    )
    parser.add_argument(
        "--better_model",
        type=str,
        default="ttfm",
        help="Model that should have lower MAE (the one we want to show is better)",
    )
    parser.add_argument(
        "--worse_model",
        type=str,
        default="chronos_univar",
        help="Model that should have higher MAE (the baseline to compare against)",
    )
    # Keep old args for backward compatibility
    parser.add_argument(
        "--ttfm_model",
        type=str,
        default=None,
        help="(Deprecated) Use --better_model instead",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="(Deprecated) Use --worse_model instead",
    )
    parser.add_argument(
        "--models_to_plot",
        type=str,
        default="ttfm,chronos_univar",
        help="Comma-separated list of models to plot",
    )
    parser.add_argument(
        "--per_dataset",
        action="store_true",
        default=True,
        help="Select top_k samples per dataset (default: True)",
    )
    parser.add_argument(
        "--global_selection",
        action="store_true",
        help="Select top_k samples globally instead of per-dataset",
    )
    parser.add_argument(
        "--create_grid",
        action="store_true",
        help="Create grid plots with multiple samples per figure",
    )
    parser.add_argument(
        "--grid_only",
        action="store_true",
        help="Only create grid plots, skip individual sample plots",
    )
    parser.add_argument(
        "--grid_samples", type=int, default=10, help="Number of samples per grid plot"
    )
    parser.add_argument(
        "--show_worst",
        action="store_true",
        help="Show samples where better_model is WORST (for analysis)",
    )
    parser.add_argument(
        "--absolute_best",
        action="store_true",
        help="Show samples with lowest absolute MAE for better_model (no comparison)",
    )
    parser.add_argument(
        "--use_dates", action="store_true", help="Use actual dates from CSV on x-axis"
    )
    parser.add_argument(
        "--generate_summaries",
        action="store_true",
        help="Generate context summaries using LLM and add below plots",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="http://localhost:8004/v1",
        help="vLLM server URL for context summarization",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model name for context summarization",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    datasets_dir = Path(args.datasets_dir)

    # Handle backward compatibility for deprecated args
    better_model = args.ttfm_model if args.ttfm_model else args.better_model
    worse_model = args.baseline_model if args.baseline_model else args.worse_model

    # Auto-detect context length from directory name
    if args.context_len is None:
        import re

        match = re.search(r"context_(\d+)", results_dir.name)
        if match:
            args.context_len = int(match.group(1))
            print(f"Auto-detected context length: {args.context_len}")
        else:
            raise ValueError(
                "Could not auto-detect context_len. Please provide --context_len"
            )

    # Set output directory
    if args.output_dir is None:
        if args.absolute_best:
            suffix = f"_absolute_best_{better_model}"
        elif args.show_worst:
            suffix = "_worst"
        else:
            suffix = ""
        output_dir = results_dir / f"qualitative_plots{suffix}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse models to plot
    models_to_plot = [m.strip() for m in args.models_to_plot.split(",")]

    print("=" * 70)
    print("QUALITATIVE FORECAST PLOT GENERATOR")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Datasets directory: {datasets_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Context length: {args.context_len}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Top K samples: {args.top_k}")
    if args.absolute_best:
        print(f"Mode: Absolute best (lowest MAE for {better_model})")
    elif args.show_worst:
        print(f"Mode: Show worst ({better_model} loses to {worse_model})")
    else:
        print(f"Mode: Comparison ({better_model} beats {worse_model})")
    print(f"Models to plot: {models_to_plot}")
    print(f"Per-dataset selection: {not args.global_selection}")
    print(f"Use dates on x-axis: {args.use_dates}")
    print(f"Generate summaries: {args.generate_summaries}")

    # Load per-dataset metrics (needed for both formats); evaluation writes stats_Context_*_allsamples.csv
    per_dataset_csv = results_dir / "per_dataset_metrics.csv"
    if not per_dataset_csv.exists():
        candidates = list(Path(results_dir).glob("stats_Context_*_allsamples.csv"))
        if candidates:
            per_dataset_csv = candidates[0]
        else:
            print(
                f"Error: per_dataset_metrics.csv not found and no stats_Context_*_allsamples.csv in {results_dir}"
            )
            return

    # Load predictions - detect new vs old format
    outputs_dir = results_dir / "outputs"
    use_new_format = outputs_dir.exists() and outputs_dir.is_dir()

    if use_new_format:
        print("\nDetected new per-dataset output format...")
        predictions, gt, input_data = load_predictions_from_outputs(
            results_dir, per_dataset_csv
        )
    else:
        print("\nUsing legacy single-file format...")
        predictions = load_predictions(results_dir)

        if not predictions:
            print("Error: No predictions found")
            return

        # Load ground truth and input
        gt = np.load(results_dir / "gt.npy")
        input_data = np.load(results_dir / "input.npy")
        print(f"Ground truth shape: {gt.shape}")
        print(f"Input shape: {input_data.shape}")

    if not predictions:
        print("Error: No predictions found")
        return

    # Compute normalization parameters
    print("\nComputing normalization parameters...")
    history_means, history_stds, dataset_names = compute_raw_mean_std(
        per_dataset_csv, datasets_dir, args.context_len, args.pred_len
    )

    # Verify shapes match
    if len(history_means) != len(gt):
        print(f"Warning: Shape mismatch - means: {len(history_means)}, gt: {len(gt)}")
        # Truncate or pad as needed
        min_len = min(len(history_means), len(gt))
        history_means = history_means[:min_len]
        history_stds = history_stds[:min_len]
        dataset_names = dataset_names[:min_len]
        gt = gt[:min_len]
        input_data = input_data[:min_len]
        predictions = {k: v[:min_len] for k, v in predictions.items()}

    # Find best samples
    if args.absolute_best:
        print(f"\nFinding samples with lowest absolute MAE for {better_model}...")
        best_samples = find_absolute_best_samples(
            predictions=predictions,
            gt=gt,
            input_data=input_data,
            history_means=history_means,
            history_stds=history_stds,
            dataset_names=dataset_names,
            per_dataset_csv=per_dataset_csv,
            model_name=better_model,
            top_k=args.top_k,
            per_dataset=not args.global_selection,
        )
    elif args.show_worst:
        print(
            f"\nFinding samples where {better_model} performs WORST vs {worse_model}..."
        )
        # Swap the models to find samples where "better_model" actually loses
        best_samples = find_best_samples(
            predictions=predictions,
            gt=gt,
            input_data=input_data,
            history_means=history_means,
            history_stds=history_stds,
            dataset_names=dataset_names,
            per_dataset_csv=per_dataset_csv,
            ttfm_model=worse_model,  # Swapped
            baseline_model=better_model,  # Swapped
            top_k=args.top_k,
            per_dataset=not args.global_selection,
        )
    else:
        print(f"\nFinding samples where {better_model} outperforms {worse_model}...")
        best_samples = find_best_samples(
            predictions=predictions,
            gt=gt,
            input_data=input_data,
            history_means=history_means,
            history_stds=history_stds,
            dataset_names=dataset_names,
            per_dataset_csv=per_dataset_csv,
            ttfm_model=better_model,
            baseline_model=worse_model,
            top_k=args.top_k,
            per_dataset=not args.global_selection,
        )

    print(f"\nFound {len(best_samples)} samples to plot")

    # Load dates and text if requested
    if args.use_dates or args.generate_summaries:
        print("\nLoading dates and text annotations from data files...")
        for sample in best_samples:
            data_path = _find_dataset_file(datasets_dir, sample.dataset_name)
            if data_path.exists():
                try:
                    dates, text_annotations = load_sample_dates_and_text(
                        data_path, sample.local_idx, args.context_len, args.pred_len
                    )
                    sample.dates = dates
                    sample.text_annotations = text_annotations
                except Exception as e:
                    print(
                        f"  Warning: Could not load dates/text for {sample.dataset_name} sample {sample.local_idx}: {e}"
                    )
        print(
            f"  Loaded dates/text for {sum(1 for s in best_samples if s.dates is not None)} samples"
        )

    # Generate summaries if requested
    if args.generate_summaries:
        best_samples = generate_context_summaries(
            best_samples,
            datasets_dir,
            args.context_len,
            args.pred_len,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
        )

    # Create individual plots - organized by dataset (unless grid_only is set)
    if not args.grid_only:
        print("\nGenerating individual plots...")
        dataset_dirs_created = set()
        for i, sample in enumerate(best_samples):
            # Create dataset-specific subdirectory
            dataset_subdir = output_dir / sample.dataset_name
            if sample.dataset_name not in dataset_dirs_created:
                dataset_subdir.mkdir(parents=True, exist_ok=True)
                dataset_dirs_created.add(sample.dataset_name)

            plot_filename = f"sample{sample.sample_idx:04d}.png"
            plot_path = dataset_subdir / plot_filename

            plot_single_forecast(
                sample=sample,
                models_to_plot=models_to_plot,
                output_path=plot_path,
                show_normalized=False,
                show_summary=args.generate_summaries,
            )

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(best_samples)} plots")

        print(
            f"  Generated {len(best_samples)} individual plots in {len(dataset_dirs_created)} dataset directories"
        )
    else:
        print("\nSkipping individual plots (--grid_only)")

    # Create grid plots if requested (or if grid_only is set)
    if args.create_grid or args.grid_only:
        print("\nGenerating grid plots...")

        # Group samples by dataset
        samples_by_dataset = {}
        for sample in best_samples:
            if sample.dataset_name not in samples_by_dataset:
                samples_by_dataset[sample.dataset_name] = []
            samples_by_dataset[sample.dataset_name].append(sample)

        for dataset_name, samples in samples_by_dataset.items():
            if len(samples) >= 2:
                # Save grid plot in dataset-specific directory
                dataset_subdir = output_dir / dataset_name
                dataset_subdir.mkdir(parents=True, exist_ok=True)
                grid_path = dataset_subdir / "grid.png"
                n_samples_grid = min(args.grid_samples, len(samples))

                plot_multi_sample_comparison(
                    samples=samples[:n_samples_grid],
                    models_to_plot=models_to_plot,
                    output_path=grid_path,
                )

        print(f"  Generated {len(samples_by_dataset)} grid plots")

    # Save sample metadata
    metadata = []
    for sample in best_samples:
        metadata.append(
            {
                "sample_idx": sample.sample_idx,
                "dataset_name": sample.dataset_name,
                "local_idx": sample.local_idx,
                "mae_ttfm": sample.mae_ttfm,
                "mae_baseline": sample.mae_baseline,
                "mae_advantage": sample.mae_advantage,
                "history_mean": sample.history_mean,
                "history_std": sample.history_std,
            }
        )

    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_dir / "sample_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nSample metadata saved to: {metadata_path}")

    print(f"\nAll plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
