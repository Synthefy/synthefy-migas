#!/usr/bin/env python3
"""
Robust script to compare evaluation results from multiple CSV sources.

Produces a single PDF file with a comparison table. No intermediate files are left behind.

Usage:
    python scripts/latex/compare_results.py --config scripts/latex/config_compare_example.yaml

Supports two YAML config formats:

  1) Flat (single setting):
        sources:
          - label: "Baselines"
            csv: "path/to/stats.csv"
            models: [chronos_univar, gpt_forecast]

  2) Grouped (multiple context/horizon settings):
        settings:
          - label: "H=4, Ctx=64"
            sources:
              - label: "Baselines"
                csv: "path/to/pred_4/stats.csv"
                models: [chronos_univar, gpt_forecast]
          - label: "H=16, Ctx=64"
            sources:
              - label: "Baselines"
                csv: "path/to/pred_16/stats.csv"
                models: [chronos_univar, gpt_forecast]
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


METRIC_LABELS = {
    "mean_mape": "Mean MAPE ($\\downarrow$)",
    "mean_mae": "Mean MAE ($\\downarrow$)",
    "median_mape": "Median MAPE ($\\downarrow$)",
    "median_mae": "Median MAE ($\\downarrow$)",
    "mean_mse": "Mean MSE ($\\downarrow$)",
    "median_mse": "Median MSE ($\\downarrow$)",
    "std_mae": "Std MAE",
    "std_mape": "Std MAPE",
}

DEFAULT_MODEL_NAMES = {
    "ttfm": "TTFM",
    "ttfm_timesfm": "TTFM-TimesFM",
    "timeseries": "TS-Only",
    "chronos_univar": "Chronos2",
    "chronos_multivar": "Chronos2 MV",
    "chronos_emb": "Chronos2 MV",
    "gpt_forecast": "GPT-Forecast",
    "gpt_forecast_no_text": "GPT (No Text)",
    "chronos_gpt_cov": "Chronos2+GPT",
    "chronos_gpt_dir_cov": "Chr2+GPT MD",
    "chronos_naive_cov": "Chr2+Naive",
    "timesfm_univar": "TimesFM 2.5",
    "tabpfn_ts": "TabPFN 2.5",
    "toto_univar": "Toto",
    "toto_emb": "Toto MV",
    "naive": "Naive",
    "prophet": "Prophet",
}

DEFAULT_DATASET_NAMES = {
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
    "silver_with_text": "Silver Price",
    "copper_with_text": "Copper Futures",
    "aapl_with_text": "Apple Stock",
    "msft_with_text": "Microsoft Stock",
    "tsla_with_text": "Tesla Stock",
}


def load_config(config_path: str) -> dict:
    """Load YAML config, normalise into settings-based format, resolve paths."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "metrics" not in config:
        print("Error: Missing required config key: 'metrics'")
        sys.exit(1)

    config_dir = os.path.dirname(os.path.abspath(config_path))

    if "settings" not in config:
        if "sources" not in config:
            print("Error: Config must have either 'sources' or 'settings' key.")
            sys.exit(1)
        config["settings"] = [{"label": None, "sources": config.pop("sources")}]

    for setting in config["settings"]:
        for source in setting.get("sources", []):
            csv_path = source["csv"]
            if not os.path.isabs(csv_path):
                source["csv"] = os.path.normpath(
                    os.path.join(config_dir, csv_path)
                )

    return config


def latex_escape(s: str) -> str:
    """Escape underscore for LaTeX (raw dataset names like aal_with_text break otherwise)."""
    return str(s).replace("_", "\\_")


def format_value(value, is_best: bool = False, precision: int = 3) -> str:
    """Format a numeric value, optionally bolding if best."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    formatted = f"{value:.{precision}f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def compute_elo_ratings(
    rankings: list,
    col_keys: list,
    base_rating: int = 1500,
    n_seeds: int = 32,
    k_value: int = 32,
) -> dict:
    """
    Compute ELO ratings using multielo with multiple random seeds.

    Args:
        rankings: list of lists, each inner list is [(col_key, value), ...]
                  sorted best-first (lowest value first).
        col_keys: all column keys that participate.
        base_rating: starting ELO.
        n_seeds: number of shuffled runs to average over.
        k_value: K-factor for updates.

    Returns:
        dict mapping col_key -> integer ELO rating.
    """
    try:
        from multielo import MultiElo
    except ImportError:
        print(
            "Warning: multielo not installed; skipping ELO. "
            "Install with: pip install multielo"
        )
        return {k: base_rating for k in col_keys}

    if not rankings:
        return {k: base_rating for k in col_keys}

    elo = MultiElo(k_value=k_value, d_value=400)
    all_ratings = {k: [] for k in col_keys}

    for seed in range(n_seeds):
        ratings = {k: float(base_rating) for k in col_keys}
        shuffled = list(rankings)
        random.seed(42 + seed)
        random.shuffle(shuffled)

        for ranking in shuffled:
            if len(ranking) < 2:
                continue
            result_order = [k for k, _ in ranking]
            current = np.array([ratings[k] for k in result_order])
            new = elo.get_new_ratings(current)
            for i, k in enumerate(result_order):
                ratings[k] = new[i]

        for k in col_keys:
            all_ratings[k].append(ratings[k])

    return {
        k: int(round(np.mean(all_ratings[k]))) if all_ratings[k] else base_rating
        for k in col_keys
    }


def generate_comparison_table(config: dict) -> str:
    """
    Generate a LaTeX table comparing models across sources and metrics.

    Supports multiple settings (context/horizon groups). Each setting becomes
    a visually separated block of rows. Columns are the union of all
    (source_label, model_key) pairs across settings.
    """
    metrics = config["metrics"]
    precision = config.get("precision", 3)
    caption = config.get("caption", "Comparison of evaluation results.")
    label = config.get("label", "tab:comparison")
    settings = config["settings"]

    model_names = {**DEFAULT_MODEL_NAMES, **config.get("model_display_names", {})}
    dataset_names_map = {
        **DEFAULT_DATASET_NAMES,
        **config.get("dataset_display_names", {}),
    }
    metric_labels = {**METRIC_LABELS, **config.get("metric_labels", {})}

    highlight_sources = set(config.get("highlight_sources", []))
    highlight_color = config.get("highlight_color", "EFEFEF")

    has_multiple_settings = (
        len(settings) > 1 or settings[0].get("label") is not None
    )

    settings_loaded = []
    columns = []
    col_seen = set()

    for setting in settings:
        setting_label = setting.get("label", None)
        sources_data = []

        for source in setting.get("sources", []):
            src_label = source["label"]
            src_models = source["models"]
            csv_path = source["csv"]

            if not os.path.exists(csv_path):
                print(f"Warning: CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            valid_models = []
            for model in src_models:
                missing = [
                    f"{model}_{m}"
                    for m in metrics
                    if f"{model}_{m}" not in df.columns
                ]
                if missing:
                    print(f"Warning: Columns missing in {csv_path}: {missing}")
                valid_models.append(model)

                key = (src_label, model)
                if key not in col_seen:
                    col_seen.add(key)
                    columns.append(
                        (src_label, model, src_label in highlight_sources)
                    )

            sources_data.append(
                {
                    "label": src_label,
                    "models": valid_models,
                    "df": df,
                    "highlight": src_label in highlight_sources,
                }
            )

        if not sources_data:
            print(f"Warning: No valid data for setting '{setting_label}'")
            continue

        filter_datasets = config.get("datasets", None)
        if filter_datasets:
            ds_list = filter_datasets
        else:
            ds_list = []
            ds_seen = set()
            for src in sources_data:
                for ds in src["df"]["dataset_name"].unique():
                    if ds not in ds_seen:
                        ds_list.append(ds)
                        ds_seen.add(ds)

        rows_data = []
        for ds_name in ds_list:
            row = {"dataset_name": ds_name, "n_eval_samples": None}

            for src in sources_data:
                ds_rows = src["df"][src["df"]["dataset_name"] == ds_name]
                if ds_rows.empty:
                    continue
                ds_row = ds_rows.iloc[0]

                if row["n_eval_samples"] is None:
                    n_eval = ds_row.get("n_eval_samples", None)
                    if pd.notna(n_eval):
                        row["n_eval_samples"] = int(n_eval)

                for model in src["models"]:
                    for metric in metrics:
                        col_name = f"{model}_{metric}"
                        val = ds_row.get(col_name, None)
                        if val is not None and pd.notna(val):
                            row[(src["label"], model, metric)] = float(val)

            has_data = any(k for k in row if isinstance(k, tuple))
            if has_data:
                rows_data.append(row)

        settings_loaded.append(
            {
                "label": setting_label,
                "rows_data": rows_data,
            }
        )

    if not settings_loaded:
        print("Error: No data loaded from any setting.")
        return ""

    n_cols = len(columns)
    all_rows_flat = [r for s in settings_loaded for r in s["rows_data"]]

    if not all_rows_flat:
        print("Error: No data rows found.")
        return ""

    total_data_cols = n_cols * len(metrics)

    if has_multiple_settings:
        col_spec = "llc" + "c" * total_data_cols
        first_cols = 3
    else:
        col_spec = "lc" + "c" * total_data_cols
        first_cols = 2

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    h1 = [""] * first_cols
    for metric in metrics:
        mlabel = metric_labels.get(
            metric, metric.replace("_", " ").title()
        )
        h1.append(f"\\multicolumn{{{n_cols}}}{{c}}{{{mlabel}}}")
    lines.append(" & ".join(h1) + " \\\\")

    rules = []
    start = first_cols + 1
    for _ in metrics:
        end = start + n_cols - 1
        rules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(rules))
    lines.append("")

    h2 = [""] * first_cols
    for _ in metrics:
        for src_label, _, hl in columns:
            if hl:
                h2.append(f"\\textbf{{{src_label}}}")
            else:
                h2.append(src_label)
    lines.append(" & ".join(h2) + " \\\\")

    if has_multiple_settings:
        h3 = ["\\textbf{Setting}", "\\textbf{Dataset}", "\\textbf{N}"]
    else:
        h3 = ["\\textbf{Dataset}", "\\textbf{N}"]

    for _ in metrics:
        for _, model_key, hl in columns:
            dname = model_names.get(model_key, model_key)
            if hl:
                h3.append(f"\\textbf{{{dname}}}")
            else:
                h3.append(dname)
    lines.append(" & ".join(h3) + " \\\\")
    lines.append("\\midrule")
    lines.append("")

    wins = {}
    for src_label, model_key, _ in columns:
        wins[(src_label, model_key)] = {m: 0 for m in metrics}

    for s_idx, setting_info in enumerate(settings_loaded):
        setting_label = setting_info["label"]
        rows_data = setting_info["rows_data"]

        first_in_group = True
        for row in rows_data:
            ds_name = row["dataset_name"]
            display_name = latex_escape(
                dataset_names_map.get(ds_name, ds_name)
            )
            n_str = (
                str(row["n_eval_samples"])
                if row["n_eval_samples"] is not None
                else "-"
            )

            if has_multiple_settings:
                if first_in_group and setting_label:
                    parts = [
                        f"\\textbf{{{setting_label}}}",
                        display_name,
                        n_str,
                    ]
                    first_in_group = False
                else:
                    parts = ["", display_name, n_str]
            else:
                parts = [display_name, n_str]

            for metric in metrics:
                vals = {}
                for src_label, model_key, _ in columns:
                    v = row.get((src_label, model_key, metric), None)
                    if v is not None:
                        vals[(src_label, model_key)] = v

                best_key = min(vals, key=vals.get) if vals else None
                if best_key is not None:
                    wins[best_key][metric] += 1

                for src_label, model_key, hl in columns:
                    v = row.get((src_label, model_key, metric), None)
                    is_best = best_key == (src_label, model_key)
                    fmt = format_value(v, is_best, precision)
                    if hl and v is not None:
                        fmt = (
                            f"\\cellcolor[HTML]{{{highlight_color}}}{{{fmt}}}"
                        )
                    parts.append(fmt)

            lines.append(" & ".join(parts) + " \\\\")

        if s_idx < len(settings_loaded) - 1:
            lines.append("\\midrule")
            lines.append("")

    lines.append("")
    lines.append("\\midrule")

    empty_prefix = ["", ""] if not has_multiple_settings else ["", "", ""]

    w_parts = list(empty_prefix)
    w_parts[0] = "\\textbf{Wins}"
    for metric in metrics:
        metric_wins = {k: wins[k][metric] for k in wins}
        max_w = max(metric_wins.values()) if metric_wins else 0
        for src_label, model_key, _ in columns:
            w = wins[(src_label, model_key)][metric]
            if w == max_w and w > 0:
                w_parts.append(f"\\textbf{{{w}}}")
            else:
                w_parts.append(str(w))
    lines.append(" & ".join(w_parts) + " \\\\")

    m_parts = list(empty_prefix)
    m_parts[0] = "\\textbf{Mean}"
    for metric in metrics:
        col_means = {}
        for src_label, model_key, _ in columns:
            vals = [
                r.get((src_label, model_key, metric), None)
                for r in all_rows_flat
            ]
            vals = [v for v in vals if v is not None]
            col_means[(src_label, model_key)] = (
                np.mean(vals) if vals else None
            )
        best = min(
            (v for v in col_means.values() if v is not None), default=None
        )
        for src_label, model_key, _ in columns:
            v = col_means[(src_label, model_key)]
            if v is not None:
                m_parts.append(format_value(v, v == best, precision))
            else:
                m_parts.append("-")
    lines.append(" & ".join(m_parts) + " \\\\")

    wm_parts = list(empty_prefix)
    wm_parts[0] = "\\textbf{Wt. Mean}"
    for metric in metrics:
        col_wmeans = {}
        for src_label, model_key, _ in columns:
            vals, weights = [], []
            for row in all_rows_flat:
                v = row.get((src_label, model_key, metric), None)
                n = row.get("n_eval_samples", None)
                if v is not None and n is not None and n > 0:
                    vals.append(v)
                    weights.append(n)
            col_wmeans[(src_label, model_key)] = (
                np.average(vals, weights=weights) if vals else None
            )
        best = min(
            (v for v in col_wmeans.values() if v is not None), default=None
        )
        for src_label, model_key, _ in columns:
            v = col_wmeans[(src_label, model_key)]
            if v is not None:
                wm_parts.append(format_value(v, v == best, precision))
            else:
                wm_parts.append("-")
    lines.append(" & ".join(wm_parts) + " \\\\")

    col_keys = [(sl, mk) for sl, mk, _ in columns]
    elo_parts = list(empty_prefix)
    elo_parts[0] = "\\textbf{ELO}"
    for metric in metrics:
        metric_rankings = []
        for row in all_rows_flat:
            row_vals = []
            for sl, mk, _ in columns:
                v = row.get((sl, mk, metric), None)
                if v is not None:
                    row_vals.append(((sl, mk), v))
            if len(row_vals) >= 2:
                row_vals.sort(key=lambda x: x[1])
                metric_rankings.append(row_vals)

        elo_ratings = compute_elo_ratings(metric_rankings, col_keys)
        max_elo = max(elo_ratings.values()) if elo_ratings else 0
        for sl, mk, _ in columns:
            e = elo_ratings[(sl, mk)]
            if e == max_elo:
                elo_parts.append(f"\\textbf{{{e}}}")
            else:
                elo_parts.append(str(e))
    lines.append(" & ".join(elo_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}%")
    lines.append("")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def compile_to_pdf(table_content: str, output_pdf: str) -> bool:
    """
    Compile LaTeX table to PDF using pdflatex in a temp directory.
    Only the final PDF is kept; all intermediate files are cleaned up.
    """
    standalone = table_content.replace("\\begin{table*}[t]", "\\begin{table}[h]")
    standalone = standalone.replace("\\end{table*}", "\\end{table}")

    doc = r"""\documentclass[10pt]{article}
\usepackage[margin=0.3in,landscape,paperwidth=48in,paperheight=11in]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[table]{xcolor}
\usepackage{amsmath}
\usepackage{graphicx}

\begin{document}
\pagestyle{empty}
\thispagestyle{empty}

""" + standalone + r"""

\end{document}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "table.tex")
        with open(tex_path, "w") as f:
            f.write(doc)

        try:
            subprocess.run(
                ["pdflatex", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            print("Error: pdflatex not found. Install texlive or set PATH.")
            return False

        result = None
        for _ in range(2):
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    tmpdir,
                    tex_path,
                ],
                capture_output=True,
                timeout=60,
                cwd=tmpdir,
            )

        pdf_tmp = os.path.join(tmpdir, "table.pdf")
        if os.path.exists(pdf_tmp):
            os.makedirs(
                os.path.dirname(os.path.abspath(output_pdf)) or ".",
                exist_ok=True,
            )
            shutil.copy2(pdf_tmp, output_pdf)
            return True
        print("Error: PDF compilation failed.")
        if result and result.stdout:
            log = result.stdout.decode("utf-8", errors="replace")
            for line in log.splitlines():
                if line.startswith("!"):
                    print(f"  {line}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results across CSV sources. Outputs a single PDF."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    output_pdf = config.get("output", "comparison.pdf")
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isabs(output_pdf):
        output_pdf = os.path.join(config_dir, output_pdf)

    print("Generating comparison table...")
    table_content = generate_comparison_table(config)

    if not table_content:
        sys.exit(1)

    print("Compiling to PDF...")
    success = compile_to_pdf(table_content, output_pdf)

    if success:
        print(f"\nDone! Output: {output_pdf}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
