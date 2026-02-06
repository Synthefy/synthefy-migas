#!/usr/bin/env python3
"""
Script to collate evaluation results from CSV files into a LaTeX table.

Usage:
    python generate_table.py --config config.yaml
    python generate_table.py --csv path/to/file.csv --context 16 --name "Dataset Name"
"""

import argparse
import pandas as pd
import yaml
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict


# Column mappings from CSV to display names
# Two-row headers: line1 is main name, line2 is subtitle (empty string if none)

MODEL_COLUMNS = {
    "ttfm": {
        "median": "ttfm_mean_mse",
        "mean": "ttfm_mean_mae",
        "line1": "TTFM",
        "line2": "(Ours)",
        "highlight": True,  # Our model - bold in header
    },
    "ttfm_timesfm": {
        "median": "ttfm_timesfm_mean_mse",
        "mean": "ttfm_timesfm_mean_mae",
        "line1": "TTFM",
        "line2": "TimesFM",
        "highlight": True,  # Our model - bold in header
    },
    "chronos_univar": {
        "median": "chronos_univar_mean_mse",
        "mean": "chronos_univar_mean_mae",
        "line1": "Chronos2",
        "line2": "",
    },
    "chronos_emb": {
        "median": "chronos_emb_mean_mse",
        "mean": "chronos_emb_mean_mae",
        "line1": "Chronos2",
        "line2": "MV",
    },
    "timesfm_univar": {
        "median": "timesfm_univar_mean_mse",
        "mean": "timesfm_univar_mean_mae",
        "line1": "TimesFM",
        "line2": "2.5",
    },
    "gpt_forecast": {
        "median": "gpt_forecast_mean_mse",
        "mean": "gpt_forecast_mean_mae",
        "line1": "GPT-OSS",
        "line2": "Forecast",
    },
    "chronos_gpt_cov": {
        "median": "chronos_gpt_cov_mean_mse",
        "mean": "chronos_gpt_cov_mean_mae",
        "line1": "Chronos2",
        "line2": "GPT",
    },
    "chronos_gpt_dir_cov": {
        "median": "chronos_gpt_dir_cov_mean_mse",
        "mean": "chronos_gpt_dir_cov_mean_mae",
        "line1": "Chronos2",
        "line2": "GPT MD",
    },
    "tabpfn_ts": {
        "median": "tabpfn_ts_mean_mse",
        "mean": "tabpfn_ts_mean_mae",
        "line1": "TabPFN",
        "line2": "2.5",
    },
    "prophet": {
        "median": "prophet_mean_mse",
        "mean": "prophet_mean_mae",
        "line1": "Prophet",
        "line2": "",
    },
    "toto_univar": {
        "median": "toto_univar_mean_mse",
        "mean": "toto_univar_mean_mae",
        "line1": "Toto",
        "line2": "",
    },
    "toto_emb": {
        "median": "toto_emb_mean_mse",
        "mean": "toto_emb_mean_mae",
        "line1": "Toto",
        "line2": "MV",
    },
}

# Dataset name mappings
DATASET_DISPLAY_NAMES = {
    "OIL_fred_with_text": "Crude Oil Price",
    "SP500_fred_with_text": "S\\&P500 Index",
    "elec_with_text": "Electricity Price",
    "gold_price_with_text": "Gold Price",
    "nvda_price_with_text": "NVIDIA Stock Price",
    "pjm_lmp_with_text": "PJM LMP",
    "timemmd_Economy": "TimeMMD (Economy)",
    "timemmd_Agriculture": "TimeMMD (Agriculture)",
    "timemmd_Energy": "TimeMMD (Energy)",
    "timemmd_SocialGood": "TimeMMD (SocialGood)",
}

# Set publication-quality matplotlib defaults (ICML standard)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
mpl.rcParams["font.size"] = 13
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["axes.titlesize"] = 22
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["legend.fontsize"] = 13
mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["axes.linewidth"] = 1.0
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["lines.linewidth"] = 1.5
mpl.rcParams["lines.markersize"] = 6


def load_csv_data(csv_path: str, datasets: list = None) -> pd.DataFrame:
    """Load CSV and optionally filter by dataset names."""
    df = pd.read_csv(csv_path)
    if datasets:
        df = df[df["dataset_name"].isin(datasets)]
    return df


def infer_models_from_csv(csv_path: str) -> list:
    """Infer model names from CSV columns (e.g. ttfm_mean_mae -> ttfm)."""
    df = pd.read_csv(csv_path, nrows=0)
    models = []
    for c in df.columns:
        if c.endswith("_mean_mae"):
            name = c[: -len("_mean_mae")]
            if name and name not in models:
                models.append(name)
    return models


def resolve_config_paths(config: dict, config_path: str) -> None:
    """Resolve relative csv_path in context_groups and horizon_groups relative to config file. Modifies config in place."""
    config_dir = Path(config_path).resolve().parent
    for ctx_group in config.get("context_groups", []):
        p = ctx_group.get("csv_path")
        if p and not os.path.isabs(p):
            ctx_group["csv_path"] = str((config_dir / p).resolve())
    for h_group in config.get("horizon_groups", []):
        for ctx_group in h_group.get("context_groups", []):
            p = ctx_group.get("csv_path")
            if p and not os.path.isabs(p):
                ctx_group["csv_path"] = str((config_dir / p).resolve())


def format_value(value: float, is_best: bool = False, precision: int = 3) -> str:
    """Format a numeric value, optionally bolding if best."""
    formatted = f"{value:.{precision}f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def get_best_model_for_row(row: pd.Series, models: list, metric_type: str) -> str:
    """Find the model with the lowest MAE for a given row."""
    best_model = None
    best_value = float("inf")

    for model in models:
        col = MODEL_COLUMNS[model][metric_type]
        if col in row and pd.notna(row[col]):
            if row[col] < best_value:
                best_value = row[col]
                best_model = model

    return best_model


def get_ranking_from_row(row: pd.Series, models: list, metric_type: str) -> list:
    """
    Get ranking of models for a single row (lower metric = better rank).

    Args:
        row: DataFrame row with model metrics
        models: List of model names
        metric_type: 'mean' or 'median'

    Returns:
        List of (model, value) tuples sorted by value (best first), or empty if no valid data
    """
    values = []
    for model in models:
        col = MODEL_COLUMNS[model][metric_type]
        if col in row and pd.notna(row[col]):
            values.append((model, row[col]))

    if len(values) < 2:
        return []

    # Sort by value (lower is better = rank 1)
    values.sort(key=lambda x: x[1])
    return values


def compute_elo_ratings_multielo(
    rankings: list,
    models: list,
    base_rating: int = 1500,
    n_seeds: int = 32,
    k_value: int = 32,
) -> dict:
    """
    Compute ELO ratings using multielo library with multiple random seeds.

    Each ranking is a single N-player matchup where models are ranked by metric.
    We run multiple times with shuffled matchup order and average the results.

    Args:
        rankings: List of rankings, each is [(model, value), ...] sorted best-first
        models: List of all model names
        base_rating: Starting rating (default 1500)
        n_seeds: Number of random seeds to average over (default 10)
        k_value: K-factor for ELO updates (default 32)

    Returns:
        dict: {model_name: elo_rating, ...}
    """
    import random
    from multielo import MultiElo

    if not rankings:
        return {model: base_rating for model in models}

    # Initialize multielo with linear score function
    elo = MultiElo(k_value=k_value, d_value=400)

    # Run multiple times with different orderings
    all_ratings = {model: [] for model in models}

    for seed in range(n_seeds):
        # Initialize ratings for this run
        ratings = {model: float(base_rating) for model in models}

        # Shuffle the order of matchups
        shuffled_rankings = list(rankings)
        random.seed(42 + seed)
        random.shuffle(shuffled_rankings)

        for ranking in shuffled_rankings:
            if len(ranking) < 2:
                continue

            # Get current ratings in result order (1st place first)
            result_order = [m for m, _ in ranking]
            current_ratings = np.array([ratings[m] for m in result_order])

            # Get new ratings from multielo
            new_ratings = elo.get_new_ratings(current_ratings)

            # Update ratings dict
            for i, model in enumerate(result_order):
                ratings[model] = new_ratings[i]

        # Store final ratings for this seed
        for model in models:
            all_ratings[model].append(ratings[model])

    # Average across seeds
    final_ratings = {}
    for model in models:
        if all_ratings[model]:
            final_ratings[model] = int(round(np.mean(all_ratings[model])))
        else:
            final_ratings[model] = base_rating

    return final_ratings


def flatten_config(config: dict) -> tuple:
    """
    Flatten horizon_groups into context_groups format.
    Returns (context_groups, has_horizon) where has_horizon indicates if we need horizon column.
    """
    if "horizon_groups" in config:
        # New format: horizon -> context -> dataset
        flattened = []
        for h_group in config["horizon_groups"]:
            horizon = h_group["horizon"]
            for ctx_group in h_group.get("context_groups", []):
                ctx_group_copy = ctx_group.copy()
                ctx_group_copy["horizon"] = horizon
                flattened.append(ctx_group_copy)
        return flattened, True
    else:
        # Old format: context -> dataset (backward compatible)
        return config.get("context_groups", []), False


def generate_context_mean_table(config: dict, output_path: str = None) -> str:
    """
    Generate LaTeX table with metrics averaged over horizons for each context and dataset.
    """
    models = config.get(
        "models",
        ["ttfm", "chronos_univar", "chronos_emb", "timesfm_univar", "gpt_forecast"],
    )
    context_groups, has_horizon = flatten_config(config)
    precision = config.get("precision", 3)
    caption = config.get(
        "caption_context_mean",
        "Evaluation results averaged over horizons for each context and dataset.",
    )
    label = config.get("label_context_mean", "tab:results_context_mean")
    ttfm_cell_color_html = config.get("ttfm_cell_color_html", "EFEFEF")

    n_models = len(models)
    col_spec = "llc" + "c" * n_models + "|" + "c" * n_models
    first_cols = 3

    # Collect all data grouped by (context_length, dataset_name)
    from collections import defaultdict

    grouped_data = defaultdict(list)  # {(context_len, dataset_name): [rows]}
    all_dataset_names = {}

    for ctx_group in context_groups:
        csv_path = ctx_group["csv_path"]
        datasets = ctx_group.get("datasets", None)
        dataset_names = ctx_group.get("dataset_display_names", {})
        all_dataset_names.update(dataset_names)
        context_len = ctx_group["context_length"]

        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue

        df = load_csv_data(csv_path, datasets)

        if df.empty:
            print(f"Warning: No data found in {csv_path}")
            continue

        for idx, row in df.iterrows():
            dataset_name = row["dataset_name"]
            key = (context_len, dataset_name)
            grouped_data[key].append(row)

    if not grouped_data:
        print("Warning: No data found for context mean table")
        return ""

    # Average metrics over horizons for each (context, dataset) pair
    averaged_rows = []
    for (context_len, dataset_name), rows in grouped_data.items():
        # Create averaged row
        avg_row = {"dataset_name": dataset_name, "context_length": context_len}

        # Get n_eval_samples (should be the same across all rows for this dataset)
        n_eval_samples_vals = [
            row["n_eval_samples"]
            for row in rows
            if "n_eval_samples" in row and pd.notna(row["n_eval_samples"])
        ]
        if n_eval_samples_vals:
            avg_row["n_eval_samples"] = int(
                n_eval_samples_vals[0]
            )  # Take first value as they should all be the same
        else:
            avg_row["n_eval_samples"] = None

        # Average each metric column across all rows
        for model in models:
            mean_col = MODEL_COLUMNS[model]["mean"]
            median_col = MODEL_COLUMNS[model]["median"]

            # Collect all non-null values
            mean_vals = [
                row[mean_col]
                for row in rows
                if mean_col in row and pd.notna(row[mean_col])
            ]
            median_vals = [
                row[median_col]
                for row in rows
                if median_col in row and pd.notna(row[median_col])
            ]

            # Average if we have values
            if mean_vals:
                avg_row[mean_col] = np.mean(mean_vals)
            else:
                avg_row[mean_col] = np.nan

            if median_vals:
                avg_row[median_col] = np.mean(median_vals)
            else:
                avg_row[median_col] = np.nan

        averaged_rows.append(avg_row)

    # Sort by context_length, then dataset_name
    averaged_rows.sort(key=lambda x: (x["context_length"], x["dataset_name"]))

    # Generate LaTeX table
    lines = []
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    ")
    lines.append("    \\resizebox{0.95\\textwidth}{!}{%")
    lines.append("    \\sc")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header row 1: Metric types
    empty_cols = "& " * (first_cols - 1)
    lines.append(
        f"    {empty_cols}& \\multicolumn{{{n_models}}}{{c}}{{Mean MAE ($\\downarrow$)}}"
    )
    lines.append(
        f"      & \\multicolumn{{{n_models}}}{{c}}{{Mean MSE ($\\downarrow$)}} \\\\"
    )
    lines.append(
        f"    \\cmidrule(lr){{{first_cols + 1}-{first_cols + n_models}}} \\cmidrule(lr){{{first_cols + n_models + 1}-{first_cols + 2 * n_models}}}"
    )
    lines.append("    ")

    # Header row 2: Main column names
    lines.append("    Context & Dataset & N")
    for model in models:
        line1 = MODEL_COLUMNS[model]["line1"]
        is_ours = MODEL_COLUMNS[model].get("highlight", False)
        if is_ours:
            lines.append(f"    & \\textbf{{{line1}}}")
        else:
            lines.append(f"    & {line1}")
    for model in models:
        line1 = MODEL_COLUMNS[model]["line1"]
        is_ours = MODEL_COLUMNS[model].get("highlight", False)
        if is_ours:
            lines.append(f"    & \\textbf{{{line1}}}")
        else:
            lines.append(f"    & {line1}")
    lines.append("    \\\\")
    lines.append("    ")

    # Header row 3: Subtitles
    lines.append(f"    {'& ' * (first_cols - 1)}")
    for model in models:
        line2 = MODEL_COLUMNS[model]["line2"]
        lines.append(f"    & {line2}")
    for model in models:
        line2 = MODEL_COLUMNS[model]["line2"]
        lines.append(f"    & {line2}")
    lines.append("    \\\\")
    lines.append("    \\midrule")
    lines.append("    ")

    wins = {model: {"median": 0, "mean": 0} for model in models}
    rankings_mean = []
    rankings_median = []

    prev_context = None
    for idx, avg_row in enumerate(averaged_rows):
        context_len = avg_row["context_length"]
        dataset_name = avg_row["dataset_name"]

        display_name = (
            all_dataset_names.get(
                dataset_name, DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
            )
            .replace(" (TimeMMD)", "$^*$")
            .replace(" (FNSPID)", "$^{\\#}$")
        )

        # Convert to Series for compatibility with existing functions
        row_series = pd.Series(avg_row)

        best_median = get_best_model_for_row(row_series, models, "median")
        best_mean = get_best_model_for_row(row_series, models, "mean")

        if best_median:
            wins[best_median]["median"] += 1
        if best_mean:
            wins[best_mean]["mean"] += 1

        # Collect rankings for ELO
        ranking_mean = get_ranking_from_row(row_series, models, "mean")
        ranking_median = get_ranking_from_row(row_series, models, "median")
        if ranking_mean:
            rankings_mean.append(ranking_mean)
        if ranking_median:
            rankings_median.append(ranking_median)

        # Build row
        n_samples_str = (
            str(int(avg_row["n_eval_samples"]))
            if "n_eval_samples" in avg_row and pd.notna(avg_row["n_eval_samples"])
            else "-"
        )
        if prev_context != context_len:
            lines.append(
                f"    \\textbf{{{context_len}}} & {display_name} & {n_samples_str}"
            )
            prev_context = context_len
        else:
            lines.append(f"    & {display_name} & {n_samples_str}")

        mean_vals = []
        for model in models:
            col = MODEL_COLUMNS[model]["mean"]
            is_best = model == best_mean
            if col in row_series and pd.notna(row_series[col]):
                val = format_value(row_series[col], is_best, precision)
                if model in ["ttfm", "ttfm_timesfm"]:
                    val = f"\\cellcolor[HTML]{{{ttfm_cell_color_html}}}{{{val}}}"
            else:
                val = "-"
            mean_vals.append(val)

        median_vals = []
        for model in models:
            col = MODEL_COLUMNS[model]["median"]
            is_best = model == best_median
            if col in row_series and pd.notna(row_series[col]):
                val = format_value(row_series[col], is_best, precision)
                if model in ["ttfm", "ttfm_timesfm"]:
                    val = f"\\cellcolor[HTML]{{{ttfm_cell_color_html}}}{{{val}}}"
            else:
                val = "-"
            median_vals.append(val)

        lines.append(f"    & {' & '.join(mean_vals)}")
        lines.append(f"    & {' & '.join(median_vals)} \\\\")
        lines.append("    ")

        # Add midrule between different contexts
        # Check if next row has different context
        if idx + 1 < len(averaged_rows):
            next_context = averaged_rows[idx + 1]["context_length"]
            if next_context != context_len:
                lines.append("    \\midrule")
                lines.append("    ")

    # Add wins summary row
    lines.append("    \\textbf{Wins} & &")

    # Mean wins first
    max_mean_wins = max(wins[m]["mean"] for m in models)
    mean_wins = []
    for model in models:
        w = wins[model]["mean"]
        val = f"\\textbf{{{w}}}" if w == max_mean_wins else str(w)
        mean_wins.append(val)
    lines.append(f"    & {' & '.join(mean_wins)}")

    # Median wins
    max_median_wins = max(wins[m]["median"] for m in models)
    median_wins = []
    for model in models:
        w = wins[model]["median"]
        val = f"\\textbf{{{w}}}" if w == max_median_wins else str(w)
        median_wins.append(val)
    lines.append(f"    & {' & '.join(median_wins)} \\\\")

    # Compute and add ELO ratings (optional; skip if multielo not installed)
    show_elo = config.get("show_elo", True)
    if show_elo:
        try:
            elo_mean = compute_elo_ratings_multielo(rankings_mean, models)
            elo_median = compute_elo_ratings_multielo(rankings_median, models)
        except ImportError:
            show_elo = False
    if show_elo:
        # ELO row
        lines.append("    \\textbf{ELO} & &")

        # Mean ELO
        max_mean_elo = max(elo_mean[m] for m in models)
        mean_elos = []
        for model in models:
            e = elo_mean[model]
            val = f"\\textbf{{{e}}}" if e == max_mean_elo else str(e)
            mean_elos.append(val)
        lines.append(f"    & {' & '.join(mean_elos)}")

        # Median ELO
        max_median_elo = max(elo_median[m] for m in models)
        median_elos = []
        for model in models:
            e = elo_median[model]
            val = f"\\textbf{{{e}}}" if e == max_median_elo else str(e)
            median_elos.append(val)
        lines.append(f"    & {' & '.join(median_elos)} \\\\")

    # Close table
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    }")
    lines.append("    ")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("    \\end{table*}")

    latex_content = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_content)
        print(f"LaTeX context mean table written to: {output_path}")

    return latex_content


def generate_latex_table(config: dict, output_path: str = None) -> str:
    """Generate LaTeX table from configuration."""

    models = config.get(
        "models",
        ["ttfm", "chronos_univar", "chronos_emb", "timesfm_univar", "gpt_forecast"],
    )
    context_groups, has_horizon = flatten_config(config)
    precision = config.get("precision", 3)
    caption = config.get(
        "caption", "Evaluation results comparing TTFM against baseline models."
    )
    label = config.get("label", "tab:results")
    ttfm_cell_color_html = config.get("ttfm_cell_color_html", "EFEFEF")

    n_models = len(models)

    # Generate column spec - add horizon column if needed
    if has_horizon:
        col_spec = "lllc" + "c" * n_models + "|" + "c" * n_models
        first_cols = 4
    else:
        col_spec = "llc" + "c" * n_models + "|" + "c" * n_models
        first_cols = 3

    lines = []
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    ")
    lines.append("    \\resizebox{0.95\\textwidth}{!}{%")
    lines.append("    \\sc")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header row 1: Metric types
    empty_cols = "& " * (first_cols - 1)
    lines.append(
        f"    {empty_cols}& \\multicolumn{{{n_models}}}{{c}}{{Mean MAE ($\\downarrow$)}}"
    )
    lines.append(
        f"      & \\multicolumn{{{n_models}}}{{c}}{{Mean MSE ($\\downarrow$)}} \\\\"
    )
    lines.append(
        f"    \\cmidrule(lr){{{first_cols + 1}-{first_cols + n_models}}} \\cmidrule(lr){{{first_cols + n_models + 1}-{first_cols + 2 * n_models}}}"
    )
    lines.append("    ")

    # Header row 2: Main column names
    if has_horizon:
        lines.append("    Horizon & Context & Dataset & N")
    else:
        lines.append("    Context & Dataset & N")
    for model in models:
        line1 = MODEL_COLUMNS[model]["line1"]
        is_ours = MODEL_COLUMNS[model].get("highlight", False)
        if is_ours:
            lines.append(f"    & \\textbf{{{line1}}}")
        else:
            lines.append(f"    & {line1}")
    for model in models:
        line1 = MODEL_COLUMNS[model]["line1"]
        is_ours = MODEL_COLUMNS[model].get("highlight", False)
        if is_ours:
            lines.append(f"    & \\textbf{{{line1}}}")
        else:
            lines.append(f"    & {line1}")
    lines.append("    \\\\")
    lines.append("    ")

    # Header row 3: Subtitles
    lines.append(f"    {'& ' * (first_cols - 1)}")
    for model in models:
        line2 = MODEL_COLUMNS[model]["line2"]
        lines.append(f"    & {line2}")
    for model in models:
        line2 = MODEL_COLUMNS[model]["line2"]
        lines.append(f"    & {line2}")
    lines.append("    \\\\")
    lines.append("    \\midrule")
    lines.append("    ")

    wins = {model: {"median": 0, "mean": 0} for model in models}
    rankings_mean = []  # For ELO calculation (list of rankings per row)
    rankings_median = []
    total_rows = 0

    # Group by (horizon, context_length) to merge them
    from collections import OrderedDict

    merged_groups = OrderedDict()
    for ctx_group in context_groups:
        horizon = ctx_group.get("horizon", None)
        ctx_len = ctx_group["context_length"]
        key = (horizon, ctx_len)
        if key not in merged_groups:
            merged_groups[key] = []
        merged_groups[key].append(ctx_group)

    prev_horizon = None
    for (horizon, context_len), ctx_group_list in merged_groups.items():
        all_rows = []
        all_dataset_names = {}

        for ctx_group in ctx_group_list:
            csv_path = ctx_group["csv_path"]
            datasets = ctx_group.get("datasets", None)
            dataset_names = ctx_group.get("dataset_display_names", {})
            all_dataset_names.update(dataset_names)

            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found: {csv_path}")
                continue

            df = load_csv_data(csv_path, datasets)

            if df.empty:
                print(f"Warning: No data found in {csv_path}")
                continue

            for idx, row in df.iterrows():
                all_rows.append(row)

        if not all_rows:
            continue

        first_row = True
        show_horizon = has_horizon and horizon != prev_horizon

        for row in all_rows:
            total_rows += 1
            dataset_name = row["dataset_name"]

            display_name = (
                all_dataset_names.get(
                    dataset_name, DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
                )
                .replace(" (TimeMMD)", "$^*$")
                .replace(" (FNSPID)", "$^{\\#}$")
            )

            best_median = get_best_model_for_row(row, models, "median")
            best_mean = get_best_model_for_row(row, models, "mean")

            if best_median:
                wins[best_median]["median"] += 1
            if best_mean:
                wins[best_mean]["mean"] += 1

            # Collect rankings for ELO (multielo)
            ranking_mean = get_ranking_from_row(row, models, "mean")
            ranking_median = get_ranking_from_row(row, models, "median")
            if ranking_mean:
                rankings_mean.append(ranking_mean)
            if ranking_median:
                rankings_median.append(ranking_median)

            # Build row with optional horizon column
            n_samples_str = (
                str(int(row["n_eval_samples"]))
                if "n_eval_samples" in row and pd.notna(row["n_eval_samples"])
                else "-"
            )
            if has_horizon:
                if first_row:
                    h_col = f"\\textbf{{{horizon}}}" if show_horizon else ""
                    lines.append(
                        f"    {h_col} & \\textbf{{{context_len}}} & {display_name} & {n_samples_str}"
                    )
                    first_row = False
                    show_horizon = False
                else:
                    lines.append(f"    & & {display_name} & {n_samples_str}")
            else:
                if first_row:
                    lines.append(
                        f"    \\textbf{{{context_len}}} & {display_name} & {n_samples_str}"
                    )
                    first_row = False
                else:
                    lines.append(f"    & {display_name} & {n_samples_str}")

            mean_vals = []
            for model in models:
                col = MODEL_COLUMNS[model]["mean"]
                is_best = model == best_mean
                if col in row and pd.notna(row[col]):
                    val = format_value(row[col], is_best, precision)
                    if model in ["ttfm", "ttfm_timesfm"]:
                        val = f"\\cellcolor[HTML]{{{ttfm_cell_color_html}}}{{{val}}}"
                else:
                    val = "-"
                mean_vals.append(val)

            median_vals = []
            for model in models:
                col = MODEL_COLUMNS[model]["median"]
                is_best = model == best_median
                if col in row and pd.notna(row[col]):
                    val = format_value(row[col], is_best, precision)
                    if model in ["ttfm", "ttfm_timesfm"]:
                        val = f"\\cellcolor[HTML]{{{ttfm_cell_color_html}}}{{{val}}}"
                else:
                    val = "-"
                median_vals.append(val)

            lines.append(
                f"    & {' & '.join(mean_vals)} & {' & '.join(median_vals)} \\\\"
            )
            lines.append("    ")

        prev_horizon = horizon
        lines.append("    \\midrule")

    # Add wins summary row - clean format
    if has_horizon:
        lines.append("    \\textbf{Wins} & & &")
    else:
        lines.append("    \\textbf{Wins} & &")

    # Mean wins first
    max_mean_wins = max(wins[m]["mean"] for m in models)
    mean_wins = []
    for model in models:
        w = wins[model]["mean"]
        val = f"\\textbf{{{w}}}" if w == max_mean_wins else str(w)
        mean_wins.append(val)
    lines.append(f"    & {' & '.join(mean_wins)}")

    # Median wins
    max_median_wins = max(wins[m]["median"] for m in models)
    median_wins = []
    for model in models:
        w = wins[model]["median"]
        val = f"\\textbf{{{w}}}" if w == max_median_wins else str(w)
        median_wins.append(val)
    lines.append(f"    & {' & '.join(median_wins)} \\\\")

    # Compute and add ELO ratings using multielo (optional; skip if multielo not installed)
    show_elo = config.get("show_elo", True)
    if show_elo:
        try:
            elo_mean = compute_elo_ratings_multielo(rankings_mean, models)
            elo_median = compute_elo_ratings_multielo(rankings_median, models)
        except ImportError:
            show_elo = False
    if show_elo:
        # ELO row
        if has_horizon:
            lines.append("    \\textbf{ELO} & & &")
        else:
            lines.append("    \\textbf{ELO} & &")

        # Mean ELO
        max_mean_elo = max(elo_mean[m] for m in models)
        mean_elos = []
        for model in models:
            e = elo_mean[model]
            val = f"\\textbf{{{e}}}" if e == max_mean_elo else str(e)
            mean_elos.append(val)
        lines.append(f"    & {' & '.join(mean_elos)}")

        # Median ELO
        max_median_elo = max(elo_median[m] for m in models)
        median_elos = []
        for model in models:
            e = elo_median[model]
            val = f"\\textbf{{{e}}}" if e == max_median_elo else str(e)
            median_elos.append(val)
        lines.append(f"    & {' & '.join(median_elos)} \\\\")

    # Close table - clean format
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    }")
    lines.append("    ")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("    \\end{table*}")

    latex_content = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_content)
        print(f"LaTeX table written to: {output_path}")

    return latex_content


def calculate_improvement(ttfm_mae: float, chronos_mae: float) -> float:
    """Calculate improvement percentage: (chronos - ttfm) / chronos * 100."""
    if pd.isna(ttfm_mae) or pd.isna(chronos_mae) or chronos_mae <= 0:
        return None
    return (chronos_mae - ttfm_mae) / chronos_mae * 100


def collect_improvement_data(
    config: dict,
    metric: str = "median_mae",
    exclude_datasets: set = None,
    ttfm_model: str = "ttfm",
    baseline_model: str = "chronos_univar",
) -> dict:
    """
    Collect improvement data across all settings for each dataset.

    For each dataset, finds max(baseline - ttfm) over all settings (rows),
    then computes percentage improvement using the baseline value at that setting.

    Args:
        config: Configuration dictionary
        metric: Metric to use (e.g., 'median_mae', 'mean_mae', 'median_mape', 'mean_mape')
        exclude_datasets: Set of dataset names to exclude
        ttfm_model: The TTFM model variant to use (e.g., 'ttfm', 'ttfm_timesfm')
        baseline_model: The baseline model to compare against (e.g., 'chronos_univar', 'timesfm_univar')

    Returns:
        dict: {dataset_name: improvement_pct, ...}
    """
    exclude_datasets = exclude_datasets or set()
    context_groups, _ = flatten_config(config)
    all_dataset_names = {}

    # Collect all (ttfm, baseline) pairs per dataset
    dataset_values = defaultdict(list)  # {dataset: [(ttfm_val, baseline_val), ...]}

    for ctx_group in context_groups:
        csv_path = ctx_group["csv_path"]
        datasets = ctx_group.get("datasets", None)
        dataset_names = ctx_group.get("dataset_display_names", {})
        all_dataset_names.update(dataset_names)

        if not os.path.exists(csv_path):
            continue

        df = load_csv_data(csv_path, datasets)
        if df.empty:
            continue

        for idx, row in df.iterrows():
            dataset_name = row["dataset_name"]

            # Ignore excluded datasets only
            if dataset_name in exclude_datasets:
                continue

            ttfm_col = f"{ttfm_model}_{metric}"
            baseline_col = f"{baseline_model}_{metric}"
            ttfm_val = row.get(ttfm_col, None)
            baseline_val = row.get(baseline_col, None)

            if pd.notna(ttfm_val) and pd.notna(baseline_val) and baseline_val > 0:
                dataset_values[dataset_name].append((ttfm_val, baseline_val))

    # For each dataset, compute % improvement per setting and take max
    improvement_data = {}
    for dataset_name, values in dataset_values.items():
        if not values:
            continue

        # Compute percentage improvement for each setting, take max
        max_pct = None
        for ttfm_val, baseline_val in values:
            improvement_pct = (baseline_val - ttfm_val) / baseline_val * 100
            if max_pct is None or improvement_pct > max_pct:
                max_pct = improvement_pct

        if max_pct is not None:
            improvement_data[dataset_name] = max_pct

    return improvement_data, all_dataset_names


def collect_dual_improvement_data(
    config: dict, metric: str = "median_mae", exclude_datasets: set = None
) -> tuple:
    """
    Collect improvement data for both TTFM/Chronos2 and TTFM-TimesFM/TimesFM comparisons.

    Args:
        config: Configuration dictionary
        metric: Metric to use (e.g., 'median_mae', 'mean_mae')
        exclude_datasets: Set of dataset names to exclude

    Returns:
        tuple: (chronos_improvement_data, timesfm_improvement_data, all_dataset_names)
    """
    # Collect TTFM vs Chronos2 improvement
    chronos_data, all_names1 = collect_improvement_data(
        config,
        metric,
        exclude_datasets,
        ttfm_model="ttfm",
        baseline_model="chronos_univar",
    )

    # Collect TTFM-TimesFM vs TimesFM improvement
    timesfm_data, all_names2 = collect_improvement_data(
        config,
        metric,
        exclude_datasets,
        ttfm_model="ttfm_timesfm",
        baseline_model="timesfm_univar",
    )

    # Merge dataset names
    all_dataset_names = {**all_names1, **all_names2}

    return chronos_data, timesfm_data, all_dataset_names


def collect_metric_values(
    config: dict, metric: str = "median_mae", exclude_datasets: set = None
) -> dict:
    """
    Collect raw metric values for TTFM and Chronos2 across datasets.
    Uses the best (minimum) value across all context lengths for each dataset.

    Returns:
        dict: {dataset_name: {'ttfm': value, 'chronos': value}, ...}
    """
    exclude_datasets = exclude_datasets or set()
    context_groups, _ = flatten_config(config)
    metric_data = defaultdict(lambda: {"ttfm": float("inf"), "chronos": float("inf")})
    all_dataset_names = {}

    for ctx_group in context_groups:
        csv_path = ctx_group["csv_path"]
        datasets = ctx_group.get("datasets", None)
        dataset_names = ctx_group.get("dataset_display_names", {})
        all_dataset_names.update(dataset_names)

        if not os.path.exists(csv_path):
            continue

        df = load_csv_data(csv_path, datasets)
        if df.empty:
            continue

        for idx, row in df.iterrows():
            dataset_name = row["dataset_name"]
            if "timemmd" in dataset_name.lower() or dataset_name in exclude_datasets:
                continue

            ttfm_col = f"ttfm_{metric}"
            chronos_col = f"chronos_univar_{metric}"
            ttfm_val = row.get(ttfm_col, None)
            chronos_val = row.get(chronos_col, None)

            if pd.notna(ttfm_val) and ttfm_val < metric_data[dataset_name]["ttfm"]:
                metric_data[dataset_name]["ttfm"] = ttfm_val
            if (
                pd.notna(chronos_val)
                and chronos_val < metric_data[dataset_name]["chronos"]
            ):
                metric_data[dataset_name]["chronos"] = chronos_val

    # Filter out datasets with inf values
    result = {
        k: v
        for k, v in metric_data.items()
        if v["ttfm"] != float("inf") and v["chronos"] != float("inf")
    }
    return result, all_dataset_names


def plot_radar_comparison(
    config: dict,
    output_dir: Path,
    metric: str = "median_mae",
    exclude_datasets: set = None,
):
    """
    Generate ICML-style radar plot comparing TTFM vs Chronos2 across datasets.
    """
    metric_data, all_dataset_names = collect_metric_values(
        config, metric, exclude_datasets
    )

    if not metric_data:
        print("Warning: No metric data found for radar plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    datasets = list(metric_data.keys())
    display_names = [
        clean_latex_for_plot(
            all_dataset_names.get(ds, DATASET_DISPLAY_NAMES.get(ds, ds))
        )
        for ds in datasets
    ]

    ttfm_values = [metric_data[ds]["ttfm"] for ds in datasets]
    chronos_values = [metric_data[ds]["chronos"] for ds in datasets]

    # Normalize to span 0.25-0.65 range, inverted (lower error = outer = better)
    all_vals = ttfm_values + chronos_values
    min_val, max_val = min(all_vals), max(all_vals)
    val_range = max_val - min_val if max_val > min_val else 1
    # Map: min_val -> 0.65 (outer), max_val -> 0.25 (inner)
    ttfm_norm = [0.25 + 0.4 * (1 - (v - min_val) / val_range) for v in ttfm_values]
    chronos_norm = [
        0.25 + 0.4 * (1 - (v - min_val) / val_range) for v in chronos_values
    ]

    # Number of variables
    N = len(datasets)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the plot
    ttfm_norm += ttfm_norm[:1]
    chronos_norm += chronos_norm[:1]
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), dpi=300)

    # Colors - professional palette
    ttfm_color = "#2ca02c"  # Green for TTFM (ours)
    chronos_color = "#1f77b4"  # Blue for Chronos2

    # Plot data
    ax.plot(
        angles,
        ttfm_norm,
        "o-",
        linewidth=2.5,
        label="TTFM (Ours)",
        color=ttfm_color,
        markersize=8,
    )
    ax.fill(angles, ttfm_norm, alpha=0.1, color=ttfm_color)

    ax.plot(
        angles,
        chronos_norm,
        "s-",
        linewidth=2.5,
        label="Chronos2",
        color=chronos_color,
        markersize=8,
    )
    ax.fill(angles, chronos_norm, alpha=0.1, color=chronos_color)

    # Set the labels - rotated to follow the curve
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Clear default labels

    # Add rotated labels manually - split words onto separate lines, inside the circle
    for angle, label in zip(angles[:-1], display_names):
        label_multiline = label.replace(" ", "\n")
        angle_deg = np.degrees(angle)
        # Rotate text to follow the spoke, flip if on left side for readability
        if 90 < angle_deg < 270:
            rotation = angle_deg + 180
            ha = "left"
        else:
            rotation = angle_deg
            ha = "right"
        ax.text(
            angle,
            0.98,
            label_multiline,
            size=14,
            fontweight="medium",
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            linespacing=1.1,
        )

    # Style adjustments
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], size=10)  # Hide y-labels for cleaner look
    ax.grid(True, linestyle="-", alpha=0.3)

    # Legend
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.2, 1.1),
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=11,
    )

    # Title
    metric_display = metric.replace("_", " ").upper()
    ax.set_title(
        f"Model Comparison ({metric_display})\n(outer = better)",
        size=14,
        fontweight="bold",
        pad=25,
    )

    plt.tight_layout()

    output_pdf = output_dir / f"radar_{metric}.pdf"
    output_png = output_dir / f"radar_{metric}.png"

    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(output_png, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print("\nRadar plot saved to:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")


def plot_improvement_by_context(
    config: dict,
    output_dir: Path,
    metric: str = "median_mae",
    exclude_datasets: set = None,
):
    """
    Generate combined bar plot showing max TTFM improvement over Chronos for each dataset.
    """
    improvement_data, all_dataset_names = collect_improvement_data(
        config, metric, exclude_datasets
    )

    if not improvement_data:
        print("Warning: No improvement data found. Skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create combined plot showing all datasets
    plot_combined_improvement(improvement_data, all_dataset_names, output_dir, metric)


def clean_latex_for_plot(text: str) -> str:
    """Remove LaTeX escape sequences for matplotlib display."""
    # Replace math-mode superscripts first
    text = text.replace("$^*$", "*")
    text = text.replace("$^{\\#}$", "#")
    # Replace common LaTeX escapes
    text = text.replace("\\&", "&")
    text = text.replace("\\%", "%")
    text = text.replace("\\$", "$")
    text = text.replace("\\#", "#")
    text = text.replace("\\_", "_")
    text = text.replace("\\{", "{")
    text = text.replace("\\}", "}")
    return text


def wrap_text(text: str, max_length: int = 15) -> str:
    """Wrap text to multiple lines if it exceeds max_length."""
    if len(text) <= max_length:
        return text

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        # If adding this word would exceed max_length, start a new line
        if current_length + word_len + 1 > max_length and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len
        else:
            current_line.append(word)
            current_length += word_len + 1 if current_line else word_len

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def plot_combined_improvement(
    improvement_data: dict,
    all_dataset_names: dict,
    output_dir: Path,
    metric: str = "median_mae",
):
    """Create a combined plot showing max improvement for each dataset."""

    # Set Roboto Mono font
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["Roboto Mono", "DejaVu Sans Mono", "Courier New"]

    datasets = []
    max_improvements = []

    for dataset_name, improvement_pct in improvement_data.items():
        # Skip datasets with negative improvement
        if improvement_pct < 0:
            continue

        display_name = all_dataset_names.get(
            dataset_name, DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        )
        # Clean LaTeX escapes for plot display
        display_name = clean_latex_for_plot(display_name)

        datasets.append(display_name)
        max_improvements.append(improvement_pct)

    if not datasets:
        return

    # Sort by improvement (descending)
    sorted_data = sorted(
        zip(datasets, max_improvements), key=lambda x: x[1], reverse=True
    )
    datasets, max_improvements = zip(*sorted_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Calculate available space per tick to prevent overlap
    fig_width_inches = 10
    available_width_per_tick = (
        (fig_width_inches * 0.8) / len(datasets) if datasets else 1.0
    )
    max_chars_per_line = int(available_width_per_tick / 0.12)
    max_chars_per_line = max(8, min(20, max_chars_per_line))

    # Wrap dataset names for multi-line display based on available space
    wrapped_datasets = [wrap_text(ds, max_length=max_chars_per_line) for ds in datasets]

    bars = ax.bar(
        range(len(datasets)),
        max_improvements,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.8,
        width=0.7,
    )

    # Add value labels with padding above bars
    for bar, imp in zip(bars, max_improvements):
        height = bar.get_height()
        padding = (
            max(
                max(max_improvements) * 0.03,
                (max(max_improvements) - min(max_improvements)) * 0.005,
            )
            if max_improvements
            else 0.5
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + padding,
            f"{imp:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    metric_display = metric.replace("_", " ").upper()
    ax.set_ylabel("Improvement (%)", fontweight="bold", fontsize=20)
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=20)
    ax.set_title("Multimodal Gains in MAE", fontweight="normal", fontsize=20)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(wrapped_datasets, rotation=0, ha="center", fontsize=10)

    # Adjust bottom margin to accommodate multi-line labels
    max_lines = (
        max(len(label.split("\n")) for label in wrapped_datasets)
        if wrapped_datasets
        else 1
    )
    plt.subplots_adjust(bottom=0.15 + (max_lines - 1) * 0.05)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    y_max = max(max_improvements) if max_improvements else 0
    y_min = min(max_improvements) if max_improvements else 0
    ax.set_ylim(
        bottom=min(0, y_min) if y_min < 0 else 0, top=y_max * 1.15 if y_max > 0 else 1
    )

    plt.tight_layout()

    output_pdf = output_dir / f"improvement_{metric}.pdf"
    output_png = output_dir / f"improvement_{metric}.png"

    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(output_png, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print("\nCombined improvement plot saved to:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")


def compute_elo_for_plot(config: dict, metric: str = "mean_mae") -> dict:
    """
    Compute ELO ratings for all models based on the config.

    Args:
        config: Configuration dictionary
        metric: Metric to use ('mean_mae', 'median_mae', etc.)

    Returns:
        dict: {model_name: elo_rating, ...}
    """
    context_groups, _ = flatten_config(config)
    models = config.get("models", list(MODEL_COLUMNS.keys()))

    # Map metric to metric_type
    if "mape" in metric:
        metric_type = "median"
    else:
        metric_type = "mean"

    rankings = []

    # Collect all rows
    for ctx_group in context_groups:
        csv_path = ctx_group["csv_path"]
        datasets = ctx_group.get("datasets", None)

        if not os.path.exists(csv_path):
            continue

        df = load_csv_data(csv_path, datasets)
        if df.empty:
            continue

        for idx, row in df.iterrows():
            ranking = get_ranking_from_row(row, models, metric_type)
            if ranking:
                rankings.append(ranking)

    # Compute ELO using multielo (return empty dict if not installed)
    try:
        elo_ratings = compute_elo_ratings_multielo(rankings, models)
    except ImportError:
        return {}
    return elo_ratings


def plot_elo_ratings(config: dict, output_dir: Path, metric: str = "mean_mae"):
    """Create a bar plot showing ELO ratings for all models."""

    # Set Roboto Mono font
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["Roboto Mono", "DejaVu Sans Mono", "Courier New"]

    # Compute ELO ratings
    elo_ratings = compute_elo_for_plot(config, metric)

    if not elo_ratings:
        print("No ELO data to plot")
        return

    # Get display names for models
    models = list(elo_ratings.keys())
    ratings = [elo_ratings[m] for m in models]

    # Create display names (combine line1 and line2)
    display_names = []
    for model in models:
        line1 = MODEL_COLUMNS[model]["line1"]
        line2 = MODEL_COLUMNS[model].get("line2", "")
        if line2:
            display_names.append(f"{line1}\n{line2}")
        else:
            display_names.append(line1)

    # Sort by ELO rating (descending)
    sorted_data = sorted(
        zip(display_names, ratings, models), key=lambda x: x[1], reverse=True
    )
    display_names, ratings, models = zip(*sorted_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Color bars - highlight TTFM
    colors = ["#2ca02c" if m == "ttfm" else "#1f77b4" for m in models]

    bars = ax.bar(
        range(len(models)),
        ratings,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )

    # Add value labels above bars
    for bar, rating in zip(bars, ratings):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{rating}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add reference line at 1500 (base rating)
    ax.axhline(
        y=1500,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Base Rating (1500)",
        alpha=0.7,
    )

    metric_display = metric.replace("_", " ").upper()
    ax.set_ylabel("ELO Rating", fontweight="bold", fontsize=12)
    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_title(
        f"Model ELO Ratings ({metric_display})", fontweight="bold", fontsize=14
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(display_names, rotation=0, ha="center", fontsize=11)

    # Set y-axis limits
    min_rating = min(ratings)
    max_rating = max(ratings)
    y_min = min(1400, min_rating - 50)
    y_max = max_rating + 80
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=14, fontweight="bold")

    plt.tight_layout()

    output_pdf = output_dir / f"elo_ratings_{metric}.pdf"
    output_png = output_dir / f"elo_ratings_{metric}.png"

    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(output_png, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print("\nELO ratings plot saved to:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")


def plot_combined_elo_and_improvement(
    config: dict,
    output_dir: Path,
    metric: str = "mean_mae",
    exclude_datasets: set = None,
):
    """Create a combined plot with ELO ratings on top and improvement gains at bottom."""

    # Set Roboto Mono font
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["Roboto Mono", "DejaVu Sans Mono", "Courier New"]

    # Get ELO data
    elo_ratings = compute_elo_for_plot(config, metric)
    if not elo_ratings:
        print("No ELO data to plot")
        return

    # Get dual improvement data (TTFM vs Chronos2 AND TTFM-TimesFM vs TimesFM)
    chronos_improvement, timesfm_improvement, all_dataset_names = (
        collect_dual_improvement_data(config, metric, exclude_datasets)
    )

    if not chronos_improvement and not timesfm_improvement:
        print("No improvement data to plot")
        return

    # Create figure with 2 subplots stacked vertically with increased gap
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 14), dpi=300, gridspec_kw={"hspace": 0.3}
    )

    # ========== TOP PLOT: ELO Ratings ==========
    models = list(elo_ratings.keys())
    ratings = [elo_ratings[m] for m in models]

    # Create display names for models
    display_names_elo = []
    for model in models:
        line1 = MODEL_COLUMNS[model]["line1"]
        line2 = (
            MODEL_COLUMNS[model]
            .get("line2", "")
            .replace("Forecast", "")
            .replace("MV", "(MV)")
        )
        if line2:
            display_names_elo.append(f"{line1}\n{line2}")
        else:
            display_names_elo.append(line1)

    # Sort by ELO rating (descending)
    sorted_elo = sorted(
        zip(display_names_elo, ratings, models), key=lambda x: x[1], reverse=True
    )
    display_names_elo, ratings, models_sorted = zip(*sorted_elo)

    # Color bars - highlight TTFM variants in green, others in muted red
    colors_elo = [
        "#4a8c4a" if m in ["ttfm", "ttfm_timesfm"] else "#c45c5c" for m in models_sorted
    ]

    bars1 = ax1.bar(
        range(len(models_sorted)),
        ratings,
        color=colors_elo,
        edgecolor="#333333",
        linewidth=0.8,
        width=0.6,
    )

    # Add value labels above bars
    for bar, rating in zip(bars1, ratings):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{rating}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    # Add reference line at 1500 (base rating)
    ax1.axhline(
        y=1500,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Base Rating (1500)",
        alpha=0.7,
    )

    ax1.set_ylabel("ELO Rating", fontweight="bold", fontsize=22)
    ax1.set_title("Model ELO Ratings", fontweight="bold", fontsize=24)
    ax1.set_xticks(range(len(models_sorted)))
    ax1.set_xticklabels(
        display_names_elo, rotation=0, ha="center", fontsize=16, fontweight="bold"
    )
    ax1.tick_params(axis="y", labelsize=14)

    min_rating = min(ratings)
    max_rating = max(ratings)
    y_min = min(1400, min_rating - 50)
    y_max = max_rating + 80
    ax1.set_ylim(y_min, y_max)

    ax1.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, axis="y")
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", fontsize=16)

    # ========== BOTTOM PLOT: Dual Improvement Gains (Grouped Bar Chart) ==========
    # Get union of all datasets that have at least one positive improvement
    all_datasets = set()
    for dataset_name, improvement_pct in chronos_improvement.items():
        if improvement_pct >= 0:  # Include zero and positive
            all_datasets.add(dataset_name)
    for dataset_name, improvement_pct in timesfm_improvement.items():
        if improvement_pct >= 0:
            all_datasets.add(dataset_name)

    if not all_datasets:
        print("No positive improvements to plot")
        plt.close()
        return

    # Create lists for plotting
    dataset_list = sorted(all_datasets)
    chronos_vals = []
    timesfm_vals = []
    display_names = []
    dataset_names_list = []  # Keep track of original names for printing

    for dataset_name in dataset_list:
        display_name = all_dataset_names.get(
            dataset_name, DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        )
        display_name = clean_latex_for_plot(display_name)
        # Remove (TimeMMD) and (FNSPID) from display names
        display_name = display_name.replace(" (TimeMMD)", "").replace(" (FNSPID)", "")
        display_names.append(display_name)
        dataset_names_list.append(dataset_name)

        # Get improvement values (use 0 if not available or negative)
        chronos_val = chronos_improvement.get(dataset_name, 0)
        timesfm_val = timesfm_improvement.get(dataset_name, 0)
        chronos_vals.append(max(0, chronos_val) if chronos_val is not None else 0)
        timesfm_vals.append(max(0, timesfm_val) if timesfm_val is not None else 0)

    # Sort by average improvement (descending)
    avg_improvements = [(c + t) / 2 for c, t in zip(chronos_vals, timesfm_vals)]
    sorted_data = sorted(
        zip(
            display_names,
            chronos_vals,
            timesfm_vals,
            avg_improvements,
            dataset_names_list,
        ),
        key=lambda x: x[3],
        reverse=True,
    )
    display_names, chronos_vals, timesfm_vals, _, dataset_names_list = zip(*sorted_data)

    # Limit to top 15 datasets
    max_datasets = 15
    display_names = display_names[:max_datasets]
    chronos_vals = chronos_vals[:max_datasets]
    timesfm_vals = timesfm_vals[:max_datasets]
    dataset_names_list = dataset_names_list[:max_datasets]

    # Print max improvements
    max_chronos_idx = chronos_vals.index(max(chronos_vals))
    max_timesfm_idx = timesfm_vals.index(max(timesfm_vals))
    print("\n=== Max Improvements ===")
    print(
        f"TTFM vs Chronos2: {max(chronos_vals):.2f}% ({display_names[max_chronos_idx]})"
    )
    print(
        f"TTFM-TimesFM vs TimesFM: {max(timesfm_vals):.2f}% ({display_names[max_timesfm_idx]})"
    )

    # Set up grouped bar chart
    x = np.arange(len(display_names))
    bar_width = 0.35

    # Colors for the two comparisons
    chronos_color = "#4a7c9b"  # Blue for TTFM vs Chronos2
    timesfm_color = "#7a4a9b"  # Purple for TTFM-TimesFM vs TimesFM

    bars_chronos = ax2.bar(
        x - bar_width / 2,
        chronos_vals,
        bar_width,
        label="Chronos2",
        color=chronos_color,
        edgecolor="#333333",
        linewidth=0.8,
    )
    bars_timesfm = ax2.bar(
        x + bar_width / 2,
        timesfm_vals,
        bar_width,
        label="TimesFM2.5",
        color=timesfm_color,
        edgecolor="#333333",
        linewidth=0.8,
    )

    ax2.set_ylabel("Improvement (%)", fontweight="bold", fontsize=22)
    ax2.set_xlabel("Dataset", fontweight="bold", fontsize=22)
    ax2.set_title("Multimodal Gains in MAE", fontweight="bold", fontsize=24)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        display_names, rotation=45, ha="right", fontsize=18, fontweight="bold"
    )
    ax2.tick_params(axis="y", labelsize=14)
    ax2.legend(loc="upper right", fontsize=20)

    ax2.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, axis="y")
    ax2.set_axisbelow(True)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    all_improvements = list(chronos_vals) + list(timesfm_vals)
    y_max_imp = max(all_improvements) if all_improvements else 0
    ax2.set_ylim(bottom=0, top=y_max_imp * 1.15 if y_max_imp > 0 else 1)

    plt.subplots_adjust(hspace=0.4)

    output_pdf = output_dir / f"combined_elo_improvement_{metric}.pdf"
    output_png = output_dir / f"combined_elo_improvement_{metric}.png"

    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(output_png, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print("\nCombined ELO + Improvement plot saved to:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")


# Source mappings for datasets
DATASET_TARGET_SOURCES = {
    # FRED datasets
    "OIL_fred_with_text": "FRED",
    "SP500_fred_with_text": "FRED",
    # Yahoo Finance / Market data
    "elec_with_text": "FRED",
    "gold_price_with_text": "Google Finance",
    "silver_with_text": "Trading Economics",
    "copper_with_text": "Google Finance",
    "steel_with_text": "Trading Economics",
    "egg_prices_with_text": "Trading Economics",
    "aapl_with_text": "Google Finance",
    "nvda_price_with_text": "Google Finance",
    "google_with_text": "Google Finance",
    # TimeMMD datasets
    "timemmd_Agriculture_annotated": "TimeMMD",
    "timemmd_Economy_annotated": "TimeMMD",
    "timemmd_Energy_annotated": "TimeMMD",
    "timemmd_SocialGood_annotated": "TimeMMD",
    # FNSPID datasets
    "msft_with_text": "FNSPID",
    "orcl_with_text": "FNSPID",
    "cost_with_text": "FNSPID",
    "wmt_with_text": "FNSPID",
    "gs_with_text": "FNSPID",
    "jpm_with_text": "FNSPID",
    "bac_with_text": "FNSPID",
    "xom_with_text": "FNSPID",
    "tsla_with_text": "FNSPID",
    "aa_with_text": "FNSPID",
    "mcd_with_text": "FNSPID",
    "ba_with_text": "FNSPID",
}

DATASET_TEXT_SOURCES = {
    # Media Cloud text annotations
    "OIL_fred_with_text": "Media Cloud",
    "SP500_fred_with_text": "Media Cloud",
    "elec_with_text": "Media Cloud",
    "gold_price_with_text": "Media Cloud",
    "silver_with_text": "Media Cloud",
    "copper_with_text": "Media Cloud",
    "steel_with_text": "Media Cloud",
    "egg_prices_with_text": "Media Cloud",
    "aapl_with_text": "Media Cloud",
    "nvda_price_with_text": "Media Cloud",
    "google_with_text": "Media Cloud",
    # TimeMMD datasets - from TimeMMD
    "timemmd_Agriculture_annotated": "TimeMMD",
    "timemmd_Economy_annotated": "TimeMMD",
    "timemmd_Energy_annotated": "TimeMMD",
    "timemmd_SocialGood_annotated": "TimeMMD",
    # FNSPID datasets - from FNSPID corpus
    "msft_with_text": "FNSPID",
    "orcl_with_text": "FNSPID",
    "cost_with_text": "FNSPID",
    "wmt_with_text": "FNSPID",
    "gs_with_text": "FNSPID",
    "jpm_with_text": "FNSPID",
    "bac_with_text": "FNSPID",
    "xom_with_text": "FNSPID",
    "tsla_with_text": "FNSPID",
    "aa_with_text": "FNSPID",
    "mcd_with_text": "FNSPID",
    "ba_with_text": "FNSPID",
}


def generate_dataset_summary(
    config: dict, output_path: str = None, data_dir: str = None
) -> str:
    """
    Generate LaTeX table summarizing dataset statistics.

    Shows total datapoints, datapoints post-2024, target source, and text source for each dataset.

    Args:
        config: Configuration dictionary with dataset info
        output_path: Optional path to write LaTeX output
        data_dir: Directory containing the dataset CSV files

    Returns:
        str: LaTeX table content
    """
    from datetime import datetime

    # Get dataset info from config
    context_groups, _ = flatten_config(config)
    all_dataset_names = {}
    all_datasets = set()

    # Collect all datasets and their display names
    for ctx_group in context_groups:
        datasets = ctx_group.get("datasets", [])
        dataset_names = ctx_group.get("dataset_display_names", {})
        all_dataset_names.update(dataset_names)
        for ds in datasets:
            all_datasets.add(ds)

    # Also check for direct datasets list in config
    if "datasets" in config:
        for ds in config["datasets"]:
            all_datasets.add(ds)

    # Get data directory from config or use default
    if data_dir is None:
        data_dir = config.get("data_dir", "../fin_test")

    # Collect statistics for each dataset
    dataset_stats = []
    cutoff_date = datetime(2024, 6, 1)

    for dataset_name in sorted(all_datasets):
        # Try to find the CSV file
        csv_path = os.path.join(data_dir, f"{dataset_name}.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: Dataset file not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            total_points = len(df)

            # Count points post-2024 and calculate date range/frequency
            post_2024_points = 0
            date_range = "Unknown"
            frequency = "Unknown"

            if "t" in df.columns:
                # Parse dates and count those >= 2024-01-01
                df["date_parsed"] = pd.to_datetime(df["t"], errors="coerce")
                valid_dates = df["date_parsed"].dropna()

                if len(valid_dates) > 0:
                    post_2024_points = len(df[df["date_parsed"] >= cutoff_date])

                    # Calculate date range
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    date_range = (
                        f"{min_date.strftime('%Y-%m')} -- {max_date.strftime('%Y-%m')}"
                    )

                    # Detect frequency from median time difference
                    if len(valid_dates) > 1:
                        sorted_dates = valid_dates.sort_values()
                        diffs = sorted_dates.diff().dropna()
                        median_diff = diffs.median().days

                        if median_diff <= 1:
                            frequency = "Daily"
                        elif median_diff <= 7:
                            frequency = "Weekly"
                        elif median_diff <= 14:
                            frequency = "Biweekly"
                        elif median_diff <= 31:
                            frequency = "Monthly"
                        elif median_diff <= 92:
                            frequency = "Quarterly"
                        else:
                            frequency = "Yearly"

            display_name = all_dataset_names.get(
                dataset_name, DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
            )

            # Get source information
            target_source = DATASET_TARGET_SOURCES.get(dataset_name, "Unknown")
            text_source = DATASET_TEXT_SOURCES.get(dataset_name, "Unknown")

            dataset_stats.append(
                {
                    "name": dataset_name,
                    "display_name": display_name,
                    "total": total_points,
                    "post_2024": post_2024_points,
                    "target_source": target_source,
                    "text_source": text_source,
                    "frequency": frequency,
                    "date_range": date_range,
                }
            )
        except Exception as e:
            print(f"Warning: Error reading {csv_path}: {e}")
            continue

    if not dataset_stats:
        print("Warning: No dataset statistics collected")
        return ""

    # Sort by display name
    dataset_stats.sort(key=lambda x: x["display_name"])

    # Calculate totals
    total_all = sum(d["total"] for d in dataset_stats)
    total_post_2024 = sum(d["post_2024"] for d in dataset_stats)

    # Generate LaTeX table
    caption = config.get(
        "summary_caption",
        "Dataset statistics showing total datapoints and datapoints from 2024 onwards.",
    )
    label = config.get("summary_label", "tab:dataset_summary")

    lines = []
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    ")
    lines.append("    \\begin{tabular}{lcccccc}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Dataset} & \\textbf{Target Source} & \\textbf{Text Source} & \\textbf{Frequency} & \\textbf{Date Range} & \\textbf{Samples} & \\textbf{$\\geq$2024-06} \\\\"
    )
    lines.append("    \\midrule")

    for stats in dataset_stats:
        display_name = stats["display_name"]
        total = stats["total"]
        post_2024 = stats["post_2024"]
        target_source = stats["target_source"]
        text_source = stats["text_source"]
        frequency = stats["frequency"]
        date_range = stats["date_range"]

        lines.append(
            f"    {display_name} & {target_source} & {text_source} & {frequency} & {date_range} & {total:,} & {post_2024:,} \\\\"
        )

    lines.append("    \\midrule")
    lines.append(
        f"    \\textbf{{Total}} & & & & & \\textbf{{{total_all:,}}} & \\textbf{{{total_post_2024:,}}} \\\\"
    )
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    ")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("\\end{table*}")

    latex_content = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_content)
        print(f"Dataset summary table written to: {output_path}")

    return latex_content


def generate_dataset_summary_grouped(
    config: dict, output_path: str = None, data_dir: str = None
) -> str:
    """
    Generate LaTeX table summarizing dataset statistics, grouped by text source.

    Args:
        config: Configuration dictionary with dataset info
        output_path: Optional path to write LaTeX output
        data_dir: Directory containing the dataset CSV files

    Returns:
        str: LaTeX table content
    """
    from datetime import datetime

    # Get dataset info from config
    context_groups, _ = flatten_config(config)
    all_dataset_names = {}
    all_datasets = set()

    # Collect all datasets and their display names
    for ctx_group in context_groups:
        datasets = ctx_group.get("datasets", [])
        dataset_names = ctx_group.get("dataset_display_names", {})
        all_dataset_names.update(dataset_names)
        for ds in datasets:
            all_datasets.add(ds)

    # Also check for direct datasets list in config
    if "datasets" in config:
        for ds in config["datasets"]:
            all_datasets.add(ds)

    # Get data directory from config or use default
    if data_dir is None:
        data_dir = config.get("data_dir", "../fin_test")

    # Collect statistics for each dataset
    dataset_stats = []
    cutoff_date = datetime(2024, 6, 1)

    for dataset_name in sorted(all_datasets):
        # Try to find the CSV file
        csv_path = os.path.join(data_dir, f"{dataset_name}.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: Dataset file not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            total_points = len(df)

            # Count points post-2024 and calculate date range/frequency
            post_2024_points = 0
            date_range = "Unknown"
            frequency = "Unknown"

            if "t" in df.columns:
                df["date_parsed"] = pd.to_datetime(df["t"], errors="coerce")
                valid_dates = df["date_parsed"].dropna()

                if len(valid_dates) > 0:
                    post_2024_points = len(df[df["date_parsed"] >= cutoff_date])

                    # Calculate date range
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    date_range = (
                        f"{min_date.strftime('%Y-%m')} -- {max_date.strftime('%Y-%m')}"
                    )

                    # Detect frequency from median time difference
                    if len(valid_dates) > 1:
                        sorted_dates = valid_dates.sort_values()
                        diffs = sorted_dates.diff().dropna()
                        median_diff = diffs.median().days

                        if median_diff <= 1:
                            frequency = "Daily"
                        elif median_diff <= 7:
                            frequency = "Weekly"
                        elif median_diff <= 14:
                            frequency = "Biweekly"
                        elif median_diff <= 31:
                            frequency = "Monthly"
                        elif median_diff <= 92:
                            frequency = "Quarterly"
                        else:
                            frequency = "Yearly"

            display_name = all_dataset_names.get(
                dataset_name, DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
            )

            # Get source information
            target_source = DATASET_TARGET_SOURCES.get(dataset_name, "Unknown")
            text_source = DATASET_TEXT_SOURCES.get(dataset_name, "Unknown")

            dataset_stats.append(
                {
                    "name": dataset_name,
                    "display_name": display_name,
                    "total": total_points,
                    "post_2024": post_2024_points,
                    "target_source": target_source,
                    "text_source": text_source,
                    "frequency": frequency,
                    "date_range": date_range,
                }
            )
        except Exception as e:
            print(f"Warning: Error reading {csv_path}: {e}")
            continue

    if not dataset_stats:
        print("Warning: No dataset statistics collected")
        return ""

    # Group by text source
    grouped = defaultdict(list)
    for stats in dataset_stats:
        grouped[stats["text_source"]].append(stats)

    # Sort each group by display name
    for text_source in grouped:
        grouped[text_source].sort(key=lambda x: x["display_name"])

    # Define group order (customize as needed)
    group_order = ["Media Cloud", "TimeMMD", "FNSPID"]
    # Add any other groups not in the predefined order
    for text_source in grouped:
        if text_source not in group_order:
            group_order.append(text_source)

    # Calculate totals
    total_all = sum(d["total"] for d in dataset_stats)
    total_post_2024 = sum(d["post_2024"] for d in dataset_stats)

    # Generate LaTeX table
    caption = config.get(
        "summary_caption",
        "Dataset statistics grouped by text source, showing total datapoints and datapoints from June 2024 onwards.",
    )
    label = config.get("summary_label", "tab:dataset_summary_grouped")

    lines = []
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    ")
    lines.append("    \\begin{tabular}{llccccc}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Text Source} & \\textbf{Dataset} & \\textbf{Target Source} & \\textbf{Frequency} & \\textbf{Date Range} & \\textbf{Total} & \\textbf{$\\geq$2024-06} \\\\"
    )
    lines.append("    \\midrule")

    for text_source in group_order:
        if text_source not in grouped:
            continue

        stats_list = grouped[text_source]
        group_total = sum(s["total"] for s in stats_list)
        group_post_2024 = sum(s["post_2024"] for s in stats_list)

        # First row of the group shows the text source
        first = True
        for stats in stats_list:
            display_name = stats["display_name"]
            # Remove source suffixes since we're already grouped by source
            display_name = display_name.replace(" (TimeMMD)", "").replace(
                " (FNSPID)", ""
            )
            target_source = stats["target_source"]
            frequency = stats["frequency"]
            date_range = stats["date_range"]
            total = stats["total"]
            post_2024 = stats["post_2024"]

            if first:
                lines.append(
                    f"    \\textbf{{{text_source}}} & {display_name} & {target_source} & {frequency} & {date_range} & {total:,} & {post_2024:,} \\\\"
                )
                first = False
            else:
                lines.append(
                    f"    & {display_name} & {target_source} & {frequency} & {date_range} & {total:,} & {post_2024:,} \\\\"
                )

        lines.append("    \\midrule")

    # Total row
    lines.append(
        f"    \\textbf{{Total}} & & & & & \\textbf{{{total_all:,}}} & \\textbf{{{total_post_2024:,}}} \\\\"
    )
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    ")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("\\end{table*}")

    latex_content = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_content)
        print(f"Grouped dataset summary table written to: {output_path}")

    return latex_content


def create_standalone_latex(table_content: str, output_path: str):
    """Create a standalone LaTeX document for PDF preview."""
    # Replace table* with table for single-column document
    standalone_content = table_content.replace(
        "\\begin{table*}[t]", "\\begin{table}[h]"
    )
    standalone_content = standalone_content.replace("\\end{table*}", "\\end{table}")

    doc = (
        r"""\documentclass[10pt]{article}
\usepackage[margin=0.3in,landscape,paperwidth=21in,paperheight=5in]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[table]{xcolor}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{pdflscape}

\begin{document}
\pagestyle{empty}
\thispagestyle{empty}

"""
        + standalone_content
        + r"""

\end{document}
"""
    )
    with open(output_path, "w") as f:
        f.write(doc)
    print(f"Standalone LaTeX document written to: {output_path}")


def compile_latex_to_pdf(tex_path: str, pdf_path: str = None) -> bool:
    """
    Compile LaTeX file to PDF using pdflatex.

    Args:
        tex_path: Path to the .tex file
        pdf_path: Optional output PDF path (default: same as tex_path but with .pdf extension)

    Returns:
        bool: True if compilation succeeded, False otherwise
    """
    import subprocess
    import os

    if pdf_path is None:
        pdf_path = os.path.abspath(tex_path.replace(".tex", ".pdf"))
    else:
        pdf_path = os.path.abspath(pdf_path)

    # Check if pdflatex is available
    try:
        subprocess.run(
            ["pdflatex", "--version"], capture_output=True, check=True, timeout=5
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        print(f"Warning: pdflatex not found. Skipping PDF compilation for {tex_path}")
        return False

    # Compile LaTeX (run twice for proper references)
    tex_dir = os.path.dirname(os.path.abspath(tex_path)) or os.getcwd()
    tex_file = os.path.basename(tex_path)

    try:
        # First compilation
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                tex_dir,
                tex_file,
            ],
            capture_output=True,
            timeout=60,
            cwd=tex_dir,
        )

        # Second compilation (for references)
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                tex_dir,
                tex_file,
            ],
            capture_output=True,
            timeout=60,
            cwd=tex_dir,
        )

        # Clean up auxiliary files
        base_name = os.path.splitext(tex_file)[0]
        aux_files = [".aux", ".log"]
        for ext in aux_files:
            aux_path = os.path.join(tex_dir, base_name + ext)
            if os.path.exists(aux_path):
                try:
                    os.remove(aux_path)
                except OSError:
                    pass

        # Check if PDF was created
        if os.path.exists(pdf_path):
            print(f"PDF generated: {pdf_path}")
            return True
        else:
            print(f"Warning: PDF compilation may have failed for {tex_path}")
            return False

    except subprocess.TimeoutExpired:
        print(f"Warning: PDF compilation timed out for {tex_path}")
        return False
    except Exception as e:
        print(f"Warning: Error compiling PDF for {tex_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from CSV evaluation results"
    )
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--output", type=str, default="results_table.tex", help="Output LaTeX file path"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Also generate standalone LaTeX for PDF",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate improvement plots"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory for plots (default: same as output)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="median_mae",
        choices=["median_mae", "mean_mae", "median_mape", "mean_mape"],
        help="Metric for plots (default: median_mae)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated dataset names to exclude from plots",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate dataset summary table"
    )
    parser.add_argument(
        "--summary-grouped",
        action="store_true",
        help="Generate dataset summary table grouped by text source",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing dataset CSV files (for --summary)",
    )

    args = parser.parse_args()

    # Parse exclude list
    exclude_datasets = set(x.strip() for x in args.exclude.split(",") if x.strip())

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Resolve relative csv_paths relative to config file directory
        resolve_config_paths(config, args.config)
        # Auto-detect models from CSV if models is missing or 'auto'
        models_cfg = config.get("models")
        if models_cfg is None or models_cfg == "auto":
            context_groups, _ = flatten_config(config)
            first_csv = context_groups[0]["csv_path"] if context_groups else None
            if first_csv and os.path.exists(first_csv):
                inferred = infer_models_from_csv(first_csv)
                if inferred:
                    config["models"] = inferred
                    # Add default MODEL_COLUMNS for any model not already defined
                    for m in inferred:
                        if m not in MODEL_COLUMNS:
                            MODEL_COLUMNS[m] = {
                                "mean": f"{m}_mean_mae",
                                "median": f"{m}_mean_mse",
                                "line1": m.replace("_", " ").title(),
                                "line2": "",
                            }
                    print(f"Auto-detected models from CSV: {inferred}")
            elif not context_groups:
                config["models"] = config.get("models") or [
                    "ttfm",
                    "chronos_univar",
                    "timesfm_univar",
                ]
    else:
        # Default config for demonstration
        config = {
            "models": [
                "ttfm",
                "chronos_univar",
                "chronos_emb",
                "timesfm_univar",
                "gpt_forecast",
            ],
            "precision": 3,
            "caption": "\\textbf{TTFM outperforms SOTA forecasting models on held-out paired text and time series datasets.} TTFM (Ours) achieves strong performance across multiple datasets and context lengths.",
            "label": "tab:main_results",
            "context_groups": [],
        }
        print("No config provided. Please create a config.yaml file.")
        print("See config_example.yaml for reference.")
        return

    # Generate table
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    table_content = generate_latex_table(config, args.output)

    # Generate context mean table (averaged over horizons)
    context_mean_output = args.output.replace(".tex", "_context_mean.tex")
    context_mean_content = generate_context_mean_table(config, context_mean_output)

    if args.standalone:
        standalone_path = args.output.replace(".tex", "_standalone.tex")
        create_standalone_latex(table_content, standalone_path)

        # Also create standalone for context mean table
        context_mean_standalone = context_mean_output.replace(".tex", "_standalone.tex")
        create_standalone_latex(context_mean_content, context_mean_standalone)

        # Compile PDFs
        print("\nCompiling PDFs...")
        pdf_path = args.output.replace(".tex", ".pdf")
        compile_latex_to_pdf(standalone_path, pdf_path)

        context_mean_pdf = context_mean_output.replace(".tex", ".pdf")
        compile_latex_to_pdf(context_mean_standalone, context_mean_pdf)

    # Generate plots
    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating plots...")
        # plot_improvement_by_context(config, plot_dir, args.metric, exclude_datasets)
        # plot_radar_comparison(config, plot_dir, args.metric, exclude_datasets)
        # plot_elo_ratings(config, plot_dir, args.metric)
        plot_combined_elo_and_improvement(
            config, plot_dir, args.metric, exclude_datasets
        )

    # Generate dataset summary table
    if args.summary:
        summary_output = args.output.replace(".tex", "_summary.tex")
        data_dir = args.data_dir
        summary_content = generate_dataset_summary(config, summary_output, data_dir)

        if args.standalone and summary_content:
            summary_standalone = summary_output.replace(".tex", "_standalone.tex")
            create_standalone_latex(summary_content, summary_standalone)
            summary_pdf = summary_output.replace(".tex", ".pdf")
            compile_latex_to_pdf(summary_standalone, summary_pdf)

    # Generate grouped dataset summary table
    if getattr(args, "summary_grouped", False):
        grouped_output = args.output.replace(".tex", "_summary_grouped.tex")
        data_dir = args.data_dir
        grouped_content = generate_dataset_summary_grouped(
            config, grouped_output, data_dir
        )

        if args.standalone and grouped_content:
            grouped_standalone = grouped_output.replace(".tex", "_standalone.tex")
            create_standalone_latex(grouped_content, grouped_standalone)
            grouped_pdf = grouped_output.replace(".tex", ".pdf")
            compile_latex_to_pdf(grouped_standalone, grouped_pdf)


if __name__ == "__main__":
    main()
