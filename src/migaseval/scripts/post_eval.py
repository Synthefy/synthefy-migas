#!/usr/bin/env python3
"""
Single entry point for post-evaluation: bar plots, optional scatter, qualitative,
aggregate PDF, and a Markdown report.

When using --qualitative or --all, datasets_dir is read from eval_meta.json (written
by the evaluation) if not passed via --datasets_dir.

Usage:
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --scatter
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --qualitative
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --aggregate
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --all
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --qualitative --datasets_dir ./data/test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Repo root used as subprocess working directory.
_repo_root = Path(__file__).resolve().parents[3]

def discover_stats_csv(results_dir: Path) -> Path | None:
    """Find stats_Context_*_allsamples.csv in results_dir."""
    if not results_dir.is_dir():
        return None
    candidates = list(results_dir.glob("stats_Context_*_allsamples.csv"))
    return sorted(candidates)[0] if candidates else None


def get_datasets_dir_from_meta(results_dir: Path) -> str | None:
    """Read datasets_dir from eval_meta.json written by the evaluation run."""
    meta_path = results_dir / "eval_meta.json"
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("datasets_dir")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-evaluation: bar plots, optional scatter/qualitative plots, and report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing stats_Context_*_allsamples.csv and outputs/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for report and plots (default: results_dir/report)",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Also generate scatter plots (Migas-1.5 vs baseline per sample)",
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="Also generate qualitative forecast plots (requires --datasets_dir)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Generate aggregate PDF across all context lengths in the parent of results_dir",
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default=None,
        help="Directory with LLM summary JSONs for aggregate quality filtering (optional)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run bar plots, scatter, qualitative, and aggregate",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=None,
        help="Directory containing dataset CSVs for qualitative plots (optional if eval wrote eval_meta.json)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: results_dir is not a directory: {results_dir}")
        return 1

    stats_csv = discover_stats_csv(results_dir)
    if not stats_csv:
        print(f"Error: no stats_Context_*_allsamples.csv found in {results_dir}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    do_scatter = args.scatter or args.all
    do_qualitative = args.qualitative or args.all
    do_aggregate = args.aggregate or args.all
    datasets_dir = args.datasets_dir
    if do_qualitative:
        if not datasets_dir:
            datasets_dir = get_datasets_dir_from_meta(results_dir)
        if not datasets_dir:
            print(
                "Error: datasets_dir needed for qualitative plots. Either run post_eval on results from a recent "
                "evaluation (which writes eval_meta.json), or pass --datasets_dir /path/to/csvs"
            )
            return 1

    # Count datasets and models for report
    import pandas as pd

    df = pd.read_csv(stats_csv)
    n_datasets = len(df)
    model_cols_mae = [c for c in df.columns if c.endswith("_mean_mae")]
    models = [c.replace("_mean_mae", "") for c in model_cols_mae]

    # Write tables as CSV: rows = datasets, columns = models, values = mean MAE / mean MSE
    def write_metric_table_csv(df: pd.DataFrame, metric: str, out_path: Path) -> None:
        suffix = f"_mean_{metric}"
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            return
        # Column names for CSV: dataset_name + model names (without suffix)
        table_df = df[["dataset_name"] + cols].copy()
        table_df.columns = ["dataset_name"] + [c[: -len(suffix)] for c in cols]
        table_df.to_csv(out_path, index=False, float_format="%.4f", na_rep="—")

    write_metric_table_csv(df, "mae", out_dir / "table_mean_mae.csv")
    write_metric_table_csv(df, "mse", out_dir / "table_mean_mse.csv")

    report_lines = [
        "# Evaluation Report",
        "",
        f"- **Results directory**: `{results_dir}`",
        f"- **Stats file**: `{stats_csv.name}`",
        f"- **Datasets**: {n_datasets}",
        f"- **Models**: {', '.join(models)}",
        "",
        "## Outputs",
        "",
    ]

    # Tables
    report_lines.append("### Tables")
    report_lines.append("")
    report_lines.append("- [Mean MAE by dataset and model](table_mean_mae.csv)")
    report_lines.append("- [Mean MSE by dataset and model](table_mean_mse.csv)")
    report_lines.append("")

    # Bar plots (always run)
    from migaseval.scripts.plot_bars import run as run_plot_bars

    if run_plot_bars(results_dir=results_dir, out_dir=out_dir):
        report_lines.append("### Bar plots")
        report_lines.append("")
        report_lines.append(
            "- [bar_aggregate_mean_mae.png](bar_aggregate_mean_mae.png)"
        )
        report_lines.append("- [bar_grouped_mean_mae.png](bar_grouped_mean_mae.png)")
        report_lines.append("- [bar_migas15_win_pct.png](bar_migas15_win_pct.png)")
        report_lines.append("- [bar_improvement_pct.png](bar_improvement_pct.png)")
        report_lines.append(
            "- [bar_elo_mean_mae.png](bar_elo_mean_mae.png) (if multielo installed)"
        )
        report_lines.append("")
    else:
        report_lines.append("Bar plots failed or skipped.")
        report_lines.append("")

    # Scatter plots (optional)
    if do_scatter:
        cmd = [
            sys.executable,
            "-m",
            "migaseval.scripts.plot_scatter",
            "--results_dir",
            str(results_dir),
        ]
        r = subprocess.run(cmd, cwd=str(_repo_root))
        if r.returncode == 0:
            report_lines.append("### Scatter plots")
            report_lines.append("")
            report_lines.append(
                "Scatter plots were written under the results directory (e.g. `sample_scatter_plots_migas15_vs_*` and `sample_scatter_summary_*.pdf`)."
            )
            report_lines.append("")
        else:
            report_lines.append("Scatter plots failed or skipped.")
            report_lines.append("")

    # Qualitative forecast plots (optional)
    if do_qualitative:
        qual_out = out_dir / "qualitative_plots"
        qual_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "migaseval.scripts.plot_qualitative_forecasts",
            "--results_dir",
            str(results_dir),
            "--datasets_dir",
            str(datasets_dir),
            "--output_dir",
            str(qual_out),
            "--models_to_plot",
            ",".join(models),
        ]
        r = subprocess.run(cmd, cwd=str(_repo_root))
        if r.returncode == 0:
            report_lines.append("### Qualitative forecast plots")
            report_lines.append("")
            report_lines.append("- [qualitative_plots/](qualitative_plots/)")
            report_lines.append("")
        else:
            report_lines.append("Qualitative forecast plots failed or skipped.")
            report_lines.append("")

    # Aggregate PDF across context lengths (optional)
    if do_aggregate:
        from migaseval.scripts.plot_aggregate import run as run_aggregate

        # The parent of results_dir contains context_32/, context_64/, etc.
        parent_dir = results_dir.parent
        agg_pdf = out_dir / "aggregate_summary.pdf"

        # Try to find summaries_dir: explicit arg, or <parent>/summaries/
        agg_summaries = args.summaries_dir
        if not agg_summaries:
            candidate = parent_dir / "summaries"
            if candidate.is_dir():
                agg_summaries = str(candidate)

        if run_aggregate(
            output_dir=parent_dir,
            summaries_dir=agg_summaries,
            out_path=agg_pdf,
        ):
            report_lines.append("### Aggregate summary")
            report_lines.append("")
            report_lines.append(
                "- [aggregate_summary.pdf](aggregate_summary.pdf)"
            )
            report_lines.append("")
        else:
            report_lines.append("Aggregate summary failed or skipped.")
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("Generated by `python -m migaseval.scripts.post_eval`.")
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
