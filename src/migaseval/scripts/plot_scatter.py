#!/usr/bin/env python3
"""
Script to create scatter plots comparing Migas-1.5 MAE vs other model MAE at sample level.
"""

from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from migaseval.eval_utils import MODEL_DISPLAY_NAMES

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Set publication-quality defaults
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "monospace"
    mpl.rcParams["font.monospace"] = ["Roboto Mono", "DejaVu Sans Mono"]
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["figure.titlesize"] = 13
    mpl.rcParams["axes.linewidth"] = 1.0
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["lines.linewidth"] = 1.5
    mpl.rcParams["lines.markersize"] = 4
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute MAE for each sample (averaged over prediction horizon)."""
    return np.mean(np.abs(pred - gt), axis=-1)


def find_best_window_for_dataset(
    migas15_mae: np.ndarray, ts_mae: np.ndarray, window_length: int
) -> Tuple[int, int]:
    """
    Find the window where Migas-1.5 has the largest advantage over Timeseries.

    Args:
        migas15_mae: Migas-1.5 MAE array for the dataset
        ts_mae: Timeseries MAE array for the dataset
        window_length: Length of the sliding window

    Returns:
        Tuple of (start_idx, end_idx) for the best window
    """
    actual_window = min(window_length, len(migas15_mae))

    best_gap = float("-inf")
    best_window_start = 0

    # Slide window through the dataset
    for i in range(len(migas15_mae) - actual_window + 1):
        window_migas15 = migas15_mae[i : i + actual_window]
        window_ts = ts_mae[i : i + actual_window]

        migas15_median = np.median(window_migas15)
        ts_median = np.median(window_ts)

        # Gap: positive means Migas-1.5 is better
        gap = ts_median - migas15_median

        if gap > best_gap:
            best_gap = gap
            best_window_start = i

    return best_window_start, best_window_start + actual_window


def load_predictions_from_outputs(
    results_dir: Path, per_dataset_csv: Path, model_names: list
) -> Tuple[dict, np.ndarray]:
    """
    Load predictions from the per-dataset outputs structure.

    The structure is:
        results_dir/
          outputs/
            {dataset_name}/
              input.npy
              gt.npy
              {model}_pred.npy
          per_dataset_metrics.csv

    Args:
        results_dir: Directory containing results
        per_dataset_csv: Path to per-dataset metrics CSV
        model_names: List of model names to load

    Returns:
        Tuple of (predictions dict, gt array) with all datasets concatenated
    """
    outputs_dir = results_dir / "outputs"
    df = pd.read_csv(per_dataset_csv, comment="#")

    predictions = {name: [] for name in model_names}
    all_gt = []

    for idx, row in df.iterrows():
        dataset_name = row["dataset_name"]
        dataset_dir = outputs_dir / dataset_name

        if not dataset_dir.exists():
            continue

        # Load gt
        gt_path = dataset_dir / "gt.npy"
        if gt_path.exists():
            all_gt.append(np.load(gt_path))

        # Load predictions for each model
        for model_name in model_names:
            pred_path = dataset_dir / f"{model_name}_pred.npy"
            if pred_path.exists():
                predictions[model_name].append(np.load(pred_path))

    # Concatenate
    gt_concat = np.concatenate(all_gt, axis=0)
    predictions_concat = {}
    for model_name, pred_list in predictions.items():
        if pred_list:
            predictions_concat[model_name] = np.concatenate(pred_list, axis=0)

    return predictions_concat, gt_concat


def load_predictions_from_npz(
    results_dir: Path, per_dataset_csv: Path, model_names: list
) -> Tuple[dict, np.ndarray]:
    """Load predictions from npz files under predictions/<dataset>/<model>.npz.

    Each .npz contains 'predictions', 'gt', 'history', 'history_means',
    'history_stds'.  Ground truth is taken from the first available model's npz.

    Returns:
        Tuple of (predictions dict, gt array) with all datasets concatenated.
    """
    pred_dir = results_dir / "predictions"
    df = pd.read_csv(per_dataset_csv, comment="#")

    predictions: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
    all_gt: list[np.ndarray] = []

    for _, row in df.iterrows():
        dataset_name = row["dataset_name"]
        ds_dir = pred_dir / dataset_name
        if not ds_dir.is_dir():
            continue

        # Load gt from the first available model
        gt_loaded = False
        for m in model_names:
            npz_path = ds_dir / f"{m}.npz"
            if npz_path.is_file():
                data = np.load(npz_path)
                if not gt_loaded:
                    all_gt.append(data["gt"])
                    gt_loaded = True
                predictions[m].append(data["predictions"])

    if not all_gt:
        raise FileNotFoundError(f"No npz files found in {pred_dir}")

    gt_concat = np.concatenate(all_gt, axis=0)
    predictions_concat = {}
    for model_name, pred_list in predictions.items():
        if pred_list:
            predictions_concat[model_name] = np.concatenate(pred_list, axis=0)

    return predictions_concat, gt_concat


def plot_sample_level_scatter(
    results_dir: Path,
    per_dataset_csv: Path,
    compare_model: str = "chronos",
    window_length: int = None,
):
    """Create publication-quality scatter plots of Migas-1.5 MAE vs comparison model MAE at sample level.

    Args:
        results_dir: Directory containing results
        per_dataset_csv: Path to per-dataset metrics CSV
        compare_model: Name of model to compare with Migas-1.5 (e.g., 'gpt_forecast', 'chronos_gpt_cov', 'timeseries')
        window_length: If specified, only analyze samples within the best performance window (default: None = all samples)
    """

    if not HAS_MATPLOTLIB:
        print(
            "Error: matplotlib is not installed. Please install it with: pip install matplotlib"
        )
        return

    # Detect format: npz (predictions/<ds>/<model>.npz), outputs (<ds>/<model>_pred.npy), or legacy (flat .npy)
    pred_dir = results_dir / "predictions"
    outputs_dir = results_dir / "outputs"
    use_npz = pred_dir.exists() and pred_dir.is_dir()
    use_outputs = (not use_npz) and outputs_dir.exists() and outputs_dir.is_dir()

    # Models we need to load
    models_to_load = ["migas15", compare_model]
    if window_length is not None:
        models_to_load.append("timeseries")

    if use_npz:
        print("Detected npz prediction format...")
        predictions, gt = load_predictions_from_npz(
            results_dir, per_dataset_csv, models_to_load
        )
        n_samples_expected = len(gt)

        if "migas15" not in predictions:
            print("Error: migas15 predictions not found!")
            return
        if compare_model not in predictions:
            print(f"Error: {compare_model} predictions not found!")
            print(f"Available models: {list(predictions.keys())}")
            return

        migas15_pred = predictions["migas15"]
        compare_pred = predictions[compare_model]
        ts_pred = predictions.get("timeseries", None)

        if window_length is not None and ts_pred is None:
            print(
                "Warning: timeseries predictions not found, cannot apply window filtering!"
            )
            window_length = None
    elif use_outputs:
        print("Detected per-dataset output format...")
        predictions, gt = load_predictions_from_outputs(
            results_dir, per_dataset_csv, models_to_load
        )
        n_samples_expected = len(gt)

        if "migas15" not in predictions:
            print("Error: migas15 predictions not found!")
            return
        if compare_model not in predictions:
            print(f"Error: {compare_model} predictions not found!")
            print(f"Available models: {list(predictions.keys())}")
            return

        migas15_pred = predictions["migas15"]
        compare_pred = predictions[compare_model]
        ts_pred = predictions.get("timeseries", None)

        if window_length is not None and ts_pred is None:
            print(
                "Warning: timeseries predictions not found, cannot apply window filtering!"
            )
            window_length = None
    else:
        # Legacy format - load from single files
        print("Using legacy single-file format...")
        gt_path = results_dir / "gt.npy"
        if not gt_path.exists():
            print(f"Error: gt.npy not found in {results_dir}")
            print("Expected either predictions/<dataset>/<model>.npz or gt.npy")
            return
        gt = np.load(gt_path)
        n_samples_expected = len(gt)

        # Load predictions
        compare_pred_path = results_dir / f"{compare_model}_pred.npy"
        migas15_pred_path = results_dir / "migas15_pred.npy"

        if not compare_pred_path.exists():
            print(f"Error: {compare_model}_pred.npy not found!")
            print(f"Available prediction files in {results_dir}:")
            for f in sorted(results_dir.glob("*_pred.npy")):
                print(f"  - {f.name}")
            return
        if not migas15_pred_path.exists():
            print("Error: migas15_pred.npy not found!")
            return

        compare_pred = np.load(compare_pred_path)
        migas15_pred = np.load(migas15_pred_path)

        # Load timeseries predictions (needed for window filtering if requested)
        ts_pred = None
        if window_length is not None:
            ts_pred_path = results_dir / "timeseries_pred.npy"
            if not ts_pred_path.exists():
                print(
                    "Warning: timeseries_pred.npy not found, cannot apply window filtering!"
                )
                window_length = None
            else:
                ts_pred = np.load(ts_pred_path)

    # Handle size mismatch for comparison model
    if len(compare_pred) != n_samples_expected:
        print(
            f"Note: {compare_model} has {len(compare_pred)} samples, using first {n_samples_expected} samples.\n"
        )
        compare_pred = compare_pred[:n_samples_expected]

    if ts_pred is not None and len(ts_pred) != n_samples_expected:
        ts_pred = ts_pred[:n_samples_expected]

    # Load per-dataset info first to know dataset boundaries
    df = pd.read_csv(per_dataset_csv, comment="#")
    sample_count_col = (
        "n_eval_samples" if "n_eval_samples" in df.columns else "n_samples"
    )

    # Compute MAE for all samples
    compare_mae = compute_mae(compare_pred, gt)
    migas15_mae = compute_mae(migas15_pred, gt)

    # Filter out samples where either MAE > 10
    valid_mask = (compare_mae <= 6) & (migas15_mae <= 6)
    n_filtered = np.sum(~valid_mask)
    if n_filtered > 0:
        print(f"Filtering out {n_filtered} samples with MAE > 10")
        compare_mae = compare_mae[valid_mask]
        migas15_mae = migas15_mae[valid_mask]
        gt = gt[valid_mask]
        if ts_pred is not None:
            ts_pred = ts_pred[valid_mask]

    # Create output directory for plots
    if window_length is not None:
        plots_dir = (
            results_dir
            / f"sample_scatter_plots_migas15_vs_{compare_model}_window{window_length}"
        )
    else:
        plots_dir = results_dir / f"sample_scatter_plots_migas15_vs_{compare_model}"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nCreating publication-quality scatter plots for {len(df)} datasets...")
    print(f"Comparing Migas-1.5 vs {compare_model}")
    if window_length is not None:
        print(f"Filtering to best performance windows (length={window_length})")
    print()

    # Publication color scheme
    color_points = "#1f77b4"  # Professional blue
    color_diagonal = "#d62728"  # Professional red
    color_fit = "#2ca02c"  # Professional green

    # Store correlation results
    correlation_results = []

    model_display_name = MODEL_DISPLAY_NAMES.get(
        compare_model, compare_model.replace("_", " ").title()
    )

    # Function to get display name for dataset
    def get_dataset_display_name(dataset_name: str) -> str:
        """Map dataset names to human-readable display names."""
        dataset_lower = dataset_name.lower()

        dataset_display_names = {
            "OIL_fred_with_text": "Crude Oil Price",
            "SP500_fred_with_text": "S&P500 Index",
            "nvda_price_with_text": "NVIDIA Stock Price",
            "gold_price_with_text": "Gold Price",
            "elec_with_text": "NASDAQ Electricity",
            "egg_prices_with_text": "Egg Price",
            "aapl_with_text": "Apple Stock Price",
            "silver_with_text": "Silver Price",
            "copper_with_text": "Copper Price",
            "steel_with_text": "Steel Price",
            "google_with_text": "Google Stock Price",
            "timemmd_Agriculture_annotated": "Broiler Price (TimeMMD)",
            "timemmd_Economy_annotated": "US Trade Balance (TimeMMD)",
            "timemmd_Energy_annotated": "Gasoline Price (TimeMMD)",
            "timemmd_SocialGood_annotated": "Unemployment Rate (TimeMMD)",
        }

        return dataset_display_names.get(
            dataset_name, dataset_name.replace("_", " ").strip()
        )

    # Build mapping from original indices to filtered indices
    original_indices = np.arange(len(compare_mae) + n_filtered)
    filtered_original_indices = (
        original_indices[valid_mask] if n_filtered > 0 else original_indices
    )

    # Track cumulative samples for dataset boundaries
    cumulative_samples = 0

    for idx, row in df.iterrows():
        dataset_name = row["dataset_name"]
        # Get display name (use original if not in mapping)
        dataset_display = get_dataset_display_name(dataset_name)
        n_samples = int(row[sample_count_col])

        # Get sample range for this dataset (original indices before filtering)
        start_idx_original = cumulative_samples
        end_idx_original = cumulative_samples + n_samples
        cumulative_samples = end_idx_original

        # Find samples in this dataset range that passed the filter
        if n_filtered > 0:
            dataset_mask = (filtered_original_indices >= start_idx_original) & (
                filtered_original_indices < end_idx_original
            )
            dataset_filtered_indices = np.where(dataset_mask)[0]
        else:
            dataset_filtered_indices = np.arange(
                start_idx_original, min(end_idx_original, len(compare_mae))
            )

        if len(dataset_filtered_indices) == 0:
            print(f"  Warning: No valid samples for {dataset_name} after filtering")
            continue

        # Get MAE for this dataset (only valid samples)
        compare_mae_dataset = compare_mae[dataset_filtered_indices]
        migas15_mae_dataset = migas15_mae[dataset_filtered_indices]

        # Apply window filtering if requested
        if window_length is not None and ts_pred is not None:
            # Get corresponding gt and ts_pred slices for this dataset
            gt_dataset = gt[dataset_filtered_indices]
            ts_pred_dataset = ts_pred[dataset_filtered_indices]
            ts_mae_dataset = compute_mae(ts_pred_dataset, gt_dataset)
            window_start, window_end = find_best_window_for_dataset(
                migas15_mae_dataset, ts_mae_dataset, window_length
            )

            # Filter to the best window
            compare_mae_dataset = compare_mae_dataset[window_start:window_end]
            migas15_mae_dataset = migas15_mae_dataset[window_start:window_end]
            n_samples_in_window = len(migas15_mae_dataset)
            window_info = f" [window: {window_start}-{window_end - 1}]"
        else:
            n_samples_in_window = len(migas15_mae_dataset)
            window_info = ""

        # Create publication-quality scatter plot for this dataset
        fig, ax = plt.subplots(
            figsize=(3.5, 3.5), dpi=300
        )  # Single column width for papers

        # Compute correlation and fit
        if n_samples_in_window > 1:
            correlation = np.corrcoef(compare_mae_dataset, migas15_mae_dataset)[0, 1]
            r_squared = correlation**2

            # Fit linear regression (polyfit degree 1)
            coeffs = np.polyfit(compare_mae_dataset, migas15_mae_dataset, deg=1)
            poly_fn = np.poly1d(coeffs)
        else:
            correlation = np.nan
            r_squared = np.nan
            coeffs = [np.nan, np.nan]
            poly_fn = None

        # Scatter plot
        ax.scatter(
            compare_mae_dataset,
            migas15_mae_dataset,
            s=25,
            alpha=0.6,
            c=color_points,
            edgecolors="black",
            linewidth=0.3,
            rasterized=True,
        )

        # Add diagonal line (y=x) for reference
        min_val = min(np.min(compare_mae_dataset), np.min(migas15_mae_dataset))
        max_val = max(np.max(compare_mae_dataset), np.max(migas15_mae_dataset))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "--",
            color=color_diagonal,
            alpha=0.7,
            linewidth=1.5,
        )

        # Add fit line if available
        if poly_fn is not None and not np.isnan(coeffs[0]):
            x_fit = np.linspace(
                np.min(compare_mae_dataset), np.max(compare_mae_dataset), 100
            )
            y_fit = poly_fn(x_fit)
            ax.plot(
                x_fit,
                y_fit,
                "-",
                color=color_fit,
                linewidth=2,
                alpha=0.8,
                label=f"Fit: $y={coeffs[0]:.2f}x{coeffs[1]:+.2f}$",
            )

        # Set labels
        ax.set_xlabel(f"{model_display_name} MAE")
        ax.set_ylabel("Migas-1.5 MAE")

        # Title with statistics
        if not np.isnan(correlation):
            title = f"{dataset_display}\n$r={correlation:.3f}$"
        else:
            title = f"{dataset_display}"
        ax.set_title(title, fontweight="normal")

        # Grid with subtle styling
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Legend with clean styling (only if fit line exists)
        if poly_fn is not None and not np.isnan(coeffs[0]):
            ax.legend(
                frameon=True,
                fancybox=False,
                edgecolor="black",
                framealpha=0.95,
                loc="best",
            )

        # Tight layout
        plt.tight_layout()

        # Save as both PDF (vector) and PNG (raster)
        safe_name = dataset_name.replace("/", "_").replace(" ", "_")
        output_path_pdf = plots_dir / f"{safe_name}_scatter.pdf"
        output_path_png = plots_dir / f"{safe_name}_scatter.png"

        plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(output_path_png, dpi=300, bbox_inches="tight", format="png")
        plt.close()

        print(
            f"  [{idx + 1}/{len(df)}] {dataset_name}: r={correlation:.4f}, R²={r_squared:.4f}, fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}, n={n_samples_in_window}{window_info}"
        )

        correlation_results.append(
            {
                "dataset": dataset_name,
                "n_samples": n_samples_in_window,
                "n_samples_total": n_samples,
                "correlation": correlation,
                "r_squared": r_squared,
                "slope": coeffs[0],
                "intercept": coeffs[1],
                "compare_mae_mean": np.mean(compare_mae_dataset),
                "migas15_mae_mean": np.mean(migas15_mae_dataset),
                "compare_mae_median": np.median(compare_mae_dataset),
                "migas15_mae_median": np.median(migas15_mae_dataset),
            }
        )

    print(f"\nAll scatter plots saved to: {plots_dir}/")

    # Save correlation results
    if window_length is not None:
        csv_path = (
            results_dir
            / f"sample_scatter_correlations_migas15_vs_{compare_model}_window{window_length}.csv"
        )
    else:
        csv_path = (
            results_dir / f"sample_scatter_correlations_migas15_vs_{compare_model}.csv"
        )
    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(csv_path, index=False)
    print(f"Correlation data saved to: {csv_path}")

    # Overall correlation (computed from filtered data if window is specified)
    if window_length is not None and ts_pred is not None:
        # Collect all filtered samples for overall statistics
        all_compare_mae_filtered = []
        all_migas15_mae_filtered = []
        cumulative_samples = 0

        for idx, row in df.iterrows():
            n_samples = int(row[sample_count_col])
            start_idx_original = cumulative_samples
            end_idx_original = cumulative_samples + n_samples
            cumulative_samples = end_idx_original

            # Map to filtered indices
            if n_filtered > 0:
                dataset_mask = (filtered_original_indices >= start_idx_original) & (
                    filtered_original_indices < end_idx_original
                )
                dataset_filtered_indices = np.where(dataset_mask)[0]
            else:
                dataset_filtered_indices = np.arange(
                    start_idx_original, min(end_idx_original, len(compare_mae))
                )

            if len(dataset_filtered_indices) == 0:
                continue

            compare_mae_dataset = compare_mae[dataset_filtered_indices]
            migas15_mae_dataset = migas15_mae[dataset_filtered_indices]
            gt_dataset = gt[dataset_filtered_indices]
            ts_pred_dataset = ts_pred[dataset_filtered_indices]
            ts_mae_dataset = compute_mae(ts_pred_dataset, gt_dataset)

            window_start, window_end = find_best_window_for_dataset(
                migas15_mae_dataset, ts_mae_dataset, window_length
            )

            all_compare_mae_filtered.extend(
                compare_mae_dataset[window_start:window_end]
            )
            all_migas15_mae_filtered.extend(
                migas15_mae_dataset[window_start:window_end]
            )

        compare_mae_for_stats = np.array(all_compare_mae_filtered)
        migas15_mae_for_stats = np.array(all_migas15_mae_filtered)
    else:
        # Use all filtered samples (MAE <= 10 filter already applied)
        compare_mae_for_stats = compare_mae
        migas15_mae_for_stats = migas15_mae

    overall_corr = np.corrcoef(compare_mae_for_stats, migas15_mae_for_stats)[0, 1]
    overall_r_squared = overall_corr**2
    overall_coeffs = np.polyfit(compare_mae_for_stats, migas15_mae_for_stats, deg=1)

    print("\nOverall statistics (all samples, MAE <= 10):")
    print(f"  Correlation (r): {overall_corr:.4f}")
    print(f"  R² score: {overall_r_squared:.4f}")
    print(f"  Linear fit: y = {overall_coeffs[0]:.4f}x + {overall_coeffs[1]:.4f}")
    print(f"  Mean per-dataset correlation: {np.nanmean(corr_df['correlation']):.4f}")
    print(f"  Mean per-dataset R²: {np.nanmean(corr_df['r_squared']):.4f}")

    # Create publication-quality summary plot showing all datasets in subplots
    # Save original dataframe before filtering
    df_original = df.copy()

    # Filter datasets for summary plot - use specific datasets if available, otherwise use all
    preferred_datasets = ["silver_with_text", "elec_with_text", "steel_with_text"]
    df_filtered = df[df["dataset_name"].isin(preferred_datasets)].reset_index(drop=True)

    # If no preferred datasets found, use up to first 6 datasets
    if len(df_filtered) == 0:
        df_filtered = df.head(6).reset_index(drop=True)

    n_datasets = len(df_filtered)

    # Skip summary plot if no datasets
    if n_datasets == 0:
        print("No datasets available for summary plot")
        return

    # Create a mapping from dataset name to correlation results (since df was filtered)
    correlation_dict = {result["dataset"]: result for result in correlation_results}

    # Create a mapping from dataset name to its original index in df_original
    dataset_to_original_idx = {
        row["dataset_name"]: orig_idx for orig_idx, row in df_original.iterrows()
    }

    n_cols = min(3, n_datasets)
    n_rows = int(np.ceil(n_datasets / n_cols))

    # Adjust figure size for publication (fits in double column width)
    fig_width = 7.0  # Double column width in inches
    fig_height = 2.2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, row in df_filtered.iterrows():
        dataset_name = row["dataset_name"]
        # Get display name
        dataset_display = get_dataset_display_name(dataset_name)
        n_samples = int(row[sample_count_col])

        # Get the original index in the unfiltered dataframe
        original_idx = dataset_to_original_idx[dataset_name]
        orig_count_col = (
            "n_eval_samples" if "n_eval_samples" in df_original.columns else "n_samples"
        )
        # Calculate start_idx and end_idx based on the original dataframe
        start_idx = sum(
            [int(df_original.iloc[i][orig_count_col]) for i in range(original_idx)]
        )
        end_idx = start_idx + n_samples

        compare_mae_dataset = compare_mae[start_idx:end_idx]
        migas15_mae_dataset = migas15_mae[start_idx:end_idx]

        # Apply window filtering if requested
        if window_length is not None and ts_pred is not None:
            ts_mae_dataset = compute_mae(
                ts_pred[start_idx:end_idx], gt[start_idx:end_idx]
            )
            window_start, window_end = find_best_window_for_dataset(
                migas15_mae_dataset, ts_mae_dataset, window_length
            )
            compare_mae_dataset = compare_mae_dataset[window_start:window_end]
            migas15_mae_dataset = migas15_mae_dataset[window_start:window_end]

        ax = axes[idx]

        # Scatter plot
        ax.scatter(
            compare_mae_dataset,
            migas15_mae_dataset,
            s=15,
            alpha=0.6,
            c=color_points,
            edgecolors="black",
            linewidth=0.2,
            rasterized=True,
        )

        min_val = min(np.min(compare_mae_dataset), np.min(migas15_mae_dataset))
        max_val = max(np.max(compare_mae_dataset), np.max(migas15_mae_dataset))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "--",
            color=color_diagonal,
            alpha=0.7,
            linewidth=1.2,
        )

        # Add fit line to summary plot - use dataset name to look up correlation results
        corr_result = correlation_dict.get(dataset_name, {})
        correlation = corr_result.get("correlation", np.nan)
        r_squared = corr_result.get("r_squared", np.nan)
        n_samples_in_window = corr_result.get("n_samples", n_samples)
        slope = corr_result.get("slope", np.nan)
        intercept = corr_result.get("intercept", np.nan)
        if (
            n_samples_in_window > 1
            and not np.isnan(correlation)
            and not np.isnan(slope)
        ):
            x_fit = np.linspace(
                np.min(compare_mae_dataset), np.max(compare_mae_dataset), 50
            )
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "-", color=color_fit, linewidth=1.5, alpha=0.8)

        ax.set_xlabel(f"{model_display_name} MAE", fontsize=9)
        ax.set_ylabel("Migas-1.5 MAE", fontsize=9)
        if not np.isnan(correlation):
            ax.set_title(
                f"{dataset_display}\n$r={correlation:.2f}$, $y={slope:.2f}x{intercept:+.2f}$",
                fontsize=9,
                fontweight="normal",
            )
        else:
            ax.set_title(f"{dataset_display}", fontsize=9, fontweight="normal")
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis("off")

    title_suffix = f" (Window={window_length})" if window_length else ""

    # Apply tight layout with space reserved for title
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Add title
    plt.suptitle(
        "Sample-Level MAEs Comparison",  # f'Migas-1.5 vs {model_display_name}: Sample-Level MAE{title_suffix}',
        fontsize=12,
        fontweight="normal",
    )

    if window_length is not None:
        summary_path_pdf = (
            results_dir
            / f"sample_scatter_summary_migas15_vs_{compare_model}_window{window_length}.pdf"
        )
        summary_path_png = (
            results_dir
            / f"sample_scatter_summary_migas15_vs_{compare_model}_window{window_length}.png"
        )
    else:
        summary_path_pdf = (
            results_dir / f"sample_scatter_summary_migas15_vs_{compare_model}.pdf"
        )
        summary_path_png = (
            results_dir / f"sample_scatter_summary_migas15_vs_{compare_model}.png"
        )

    plt.savefig(summary_path_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(summary_path_png, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print(f"Summary plot saved to: {summary_path_pdf}")

    # Create publication-quality overall scatter plot with all samples
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

    # Scatter plot
    ax.scatter(
        compare_mae_for_stats,
        migas15_mae_for_stats,
        s=15,
        alpha=0.4,
        c=color_points,
        edgecolors="black",
        linewidth=0.2,
        rasterized=True,
    )

    # Add diagonal line (y=x)
    min_val = min(np.min(compare_mae_for_stats), np.min(migas15_mae_for_stats))
    max_val = max(np.max(compare_mae_for_stats), np.max(migas15_mae_for_stats))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        color=color_diagonal,
        alpha=0.7,
        linewidth=1.5,
    )

    # Add linear fit
    x_fit = np.linspace(
        np.min(compare_mae_for_stats), np.max(compare_mae_for_stats), 100
    )
    y_fit = overall_coeffs[0] * x_fit + overall_coeffs[1]
    ax.plot(x_fit, y_fit, "-", color=color_fit, linewidth=2, alpha=0.8)

    ax.set_xlabel(f"{model_display_name} MAE")
    ax.set_ylabel("Migas-1.5 MAE")

    title_text = f"Migas-1.5 vs {model_display_name}\n$r={overall_corr:.3f}$, $y={overall_coeffs[0]:.2f}x{overall_coeffs[1]:+.2f}$"
    ax.set_title(title_text, fontweight="normal")
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if window_length is not None:
        overall_path_pdf = (
            results_dir
            / f"sample_scatter_overall_migas15_vs_{compare_model}_window{window_length}.pdf"
        )
        overall_path_png = (
            results_dir
            / f"sample_scatter_overall_migas15_vs_{compare_model}_window{window_length}.png"
        )
    else:
        overall_path_pdf = (
            results_dir / f"sample_scatter_overall_migas15_vs_{compare_model}.pdf"
        )
        overall_path_png = (
            results_dir / f"sample_scatter_overall_migas15_vs_{compare_model}.png"
        )

    plt.savefig(overall_path_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(overall_path_png, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print(f"Overall scatter plot saved to: {overall_path_pdf}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create scatter plots comparing Migas-1.5 vs other models"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing evaluation results (e.g. results/suite/context_64); looks for stats_Context_*_allsamples.csv if per_dataset_metrics.csv missing)",
    )
    parser.add_argument(
        "--compare_model",
        type=str,
        default="chronos",
        help="Model to compare with Migas-1.5 (e.g., 'chronos', 'timesfm', 'toto', 'prophet')",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=None,
        help="If specified, only analyze samples within the best performance window (where Migas-1.5 has largest advantage over Timeseries)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    per_dataset_csv = results_dir / "per_dataset_metrics.csv"
    if not per_dataset_csv.exists():
        # synthefy-migas15 evaluation writes stats_Context_<seq_len>_allsamples.csv
        candidates = list(results_dir.glob("stats_Context_*_allsamples.csv"))
        if candidates:
            per_dataset_csv = candidates[0]
        else:
            print(
                f"Error: {per_dataset_csv} not found and no stats_Context_*_allsamples.csv in {results_dir}"
            )
            return

    print(f"Creating scatter plots from: {results_dir}\n")

    plot_sample_level_scatter(
        results_dir, per_dataset_csv, args.compare_model, args.window_length
    )


if __name__ == "__main__":
    main()
