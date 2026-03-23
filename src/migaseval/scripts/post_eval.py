#!/usr/bin/env python3
"""
Single entry point for post-evaluation: bar plots, optional scatter, qualitative,
aggregate PDF, and a Markdown report.

When using --qualitative or --all, datasets_dir is read from eval_meta.json (written
by the evaluation) if not passed via --datasets_dir.

Usage:
  # Run on a single dataset group at a specific context length
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64

  # Run across all context lengths for a single dataset group
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --aggregate

  # Run on ALL dataset groups under ./results
  uv run python -m migaseval.scripts.post_eval --results_dir ./results --aggregate

  # Other options
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --scatter
  uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --qualitative --datasets_dir ./data/test
  uv run python -m migaseval.scripts.post_eval --results_dir ./results --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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


def discover_groups(results_dir: Path) -> list[tuple[str, Path]]:
    """Discover dataset groups and their context dirs under results_dir.

    Handles three layouts:
      1. results_dir is a context dir (has stats CSV) -> [("", results_dir)]
      2. results_dir is a group dir (has context_* children) -> [("", first_context_dir)]
      3. results_dir is the root (has group dirs with context_* children) -> [(group, ctx_dir), ...]

    Returns list of (group_name, context_dir) tuples. context_dir is the first
    context dir with a stats CSV (used for bar plots / report); the parent is
    used for aggregate PDF generation.
    """
    # Case 1: results_dir itself is a context dir
    if discover_stats_csv(results_dir) is not None:
        return [("", results_dir)]

    # Case 2: results_dir has context_* children directly
    ctx_children = sorted(results_dir.glob("context_*"))
    ctx_with_csv = [c for c in ctx_children if discover_stats_csv(c) is not None]
    if ctx_with_csv:
        return [("", ctx_with_csv[0])]

    # Case 3: results_dir has group subdirs (suite, fnspid, ...) each with context_*
    groups = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        ctx_dirs = sorted(child.glob("context_*"))
        ctx_with_csv = [c for c in ctx_dirs if discover_stats_csv(c) is not None]
        if ctx_with_csv:
            groups.append((child.name, ctx_with_csv[0]))
    return groups


def run_post_eval(
    results_dir: Path,
    group_name: str,
    out_dir: Path,
    *,
    do_scatter: bool,
    do_qualitative: bool,
    do_aggregate: bool,
    datasets_dir: str | None,
    summaries_dir: str | None,
) -> list[str]:
    """Run post-eval for a single context dir. Returns report lines."""
    stats_csv = discover_stats_csv(results_dir)
    if not stats_csv:
        print(f"  No stats CSV in {results_dir}, skipping")
        return []

    label = group_name or results_dir.name
    print(f"\nUsing stats: {stats_csv}")

    import pandas as pd

    df = pd.read_csv(stats_csv, comment="#")
    n_datasets = len(df)
    model_cols_mae = [c for c in df.columns if c.endswith("_mean_mae")]
    models = [c.replace("_mean_mae", "") for c in model_cols_mae]

    print(f"Models: {models}")
    print(f"Output: {out_dir}")

    def write_metric_table_csv(df: pd.DataFrame, metric: str, out_path: Path) -> None:
        suffix = f"_mean_{metric}"
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            return
        table_df = df[["dataset_name"] + cols].copy()
        table_df.columns = ["dataset_name"] + [c[: -len(suffix)] for c in cols]
        table_df.to_csv(out_path, index=False, float_format="%.4f", na_rep="—")

    write_metric_table_csv(df, "mae", out_dir / "table_mean_mae.csv")
    write_metric_table_csv(df, "mse", out_dir / "table_mean_mse.csv")

    report_lines = [
        f"# Evaluation Report — {label}",
        "",
        f"- **Results directory**: `{results_dir}`",
        f"- **Stats file**: `{stats_csv.name}`",
        f"- **Datasets**: {n_datasets}",
        f"- **Models**: {', '.join(models)}",
        "",
        "## Outputs",
        "",
        "### Tables",
        "",
        "- [Mean MAE by dataset and model](table_mean_mae.csv)",
        "- [Mean MSE by dataset and model](table_mean_mse.csv)",
        "",
    ]

    from migaseval.scripts.plot_bars import run as run_plot_bars

    if run_plot_bars(results_dir=results_dir, out_dir=out_dir):
        report_lines += [
            "### Bar plots",
            "",
            "- [bar_aggregate_mean_mae.png](bar_aggregate_mean_mae.png)",
            "- [bar_grouped_mean_mae.png](bar_grouped_mean_mae.png)",
            "- [bar_migas15_win_pct.png](bar_migas15_win_pct.png)",
            "- [bar_improvement_pct.png](bar_improvement_pct.png)",
            "- [bar_elo_mean_mae.png](bar_elo_mean_mae.png) (if multielo installed)",
            "",
        ]
    else:
        report_lines += ["Bar plots failed or skipped.", ""]

    if do_scatter:
        cmd = [
            sys.executable, "-m", "migaseval.scripts.plot_scatter",
            "--results_dir", str(results_dir),
        ]
        r = subprocess.run(cmd, cwd=str(_repo_root))
        if r.returncode == 0:
            report_lines += [
                "### Scatter plots", "",
                "Scatter plots were written under the results directory.",
                "",
            ]
        else:
            report_lines += ["Scatter plots failed or skipped.", ""]

    if do_qualitative:
        qual_datasets_dir = datasets_dir
        if not qual_datasets_dir:
            qual_datasets_dir = get_datasets_dir_from_meta(results_dir)
        if not qual_datasets_dir:
            report_lines += [
                "Qualitative plots skipped (no datasets_dir available).", "",
            ]
        else:
            qual_out = out_dir / "qualitative_plots"
            qual_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-m", "migaseval.scripts.plot_qualitative_forecasts",
                "--results_dir", str(results_dir),
                "--datasets_dir", str(qual_datasets_dir),
                "--output_dir", str(qual_out),
                "--models_to_plot", ",".join(models),
            ]
            r = subprocess.run(cmd, cwd=str(_repo_root))
            if r.returncode == 0:
                report_lines += [
                    "### Qualitative forecast plots", "",
                    "- [qualitative_plots/](qualitative_plots/)", "",
                ]
            else:
                report_lines += ["Qualitative forecast plots failed or skipped.", ""]

    if do_aggregate:
        from migaseval.scripts.plot_aggregate import run as run_aggregate

        parent_dir = results_dir.parent
        agg_pdf = out_dir / "aggregate_summary.pdf"

        agg_summaries = summaries_dir
        if not agg_summaries:
            candidate = parent_dir / "summaries"
            if candidate.is_dir():
                agg_summaries = str(candidate)

        if run_aggregate(
            output_dir=parent_dir,
            summaries_dir=agg_summaries,
            out_path=agg_pdf,
        ):
            report_lines += [
                "### Aggregate summary", "",
                "- [aggregate_summary.pdf](aggregate_summary.pdf)", "",
            ]
        else:
            report_lines += ["Aggregate summary failed or skipped.", ""]

    return report_lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-evaluation: bar plots, optional scatter/qualitative plots, and report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Results root (e.g. ./results), a dataset group dir (e.g. ./results/suite), "
        "or a specific context dir (e.g. ./results/suite/context_64)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for report and plots (default: <context_dir>/report)",
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
        help="Generate aggregate PDF across all context lengths",
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
        help="Directory containing dataset CSVs for qualitative plots",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: results_dir is not a directory: {results_dir}")
        return 1

    groups = discover_groups(results_dir)
    if not groups:
        print(f"Error: no dataset groups with stats CSVs found in {results_dir}")
        return 1

    do_scatter = args.scatter or args.all
    do_qualitative = args.qualitative or args.all
    do_aggregate = args.aggregate or args.all

    for group_name, ctx_dir in groups:
        label = group_name or ctx_dir.name
        print(f"\n{'=' * 70}")
        print(f"  Dataset group: {label}")
        print(f"{'=' * 70}")

        if args.out_dir:
            if group_name:
                out_dir = Path(args.out_dir) / group_name
            else:
                out_dir = Path(args.out_dir)
        else:
            out_dir = ctx_dir / "report"
        out_dir.mkdir(parents=True, exist_ok=True)

        report_lines = run_post_eval(
            ctx_dir,
            label,
            out_dir,
            do_scatter=do_scatter,
            do_qualitative=do_qualitative,
            do_aggregate=do_aggregate,
            datasets_dir=args.datasets_dir,
            summaries_dir=args.summaries_dir,
        )

        if report_lines:
            report_lines += ["---", "Generated by `python -m migaseval.scripts.post_eval`."]
            report_path = out_dir / "report.md"
            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            print(f"Report written to {report_path}")

    if len(groups) > 1:
        print(f"\nProcessed {len(groups)} dataset groups: {', '.join(g[0] for g in groups)}")

        from migaseval.scripts.plot_aggregate import run_combined

        run_combined(results_root=results_dir, summaries_dir=args.summaries_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
