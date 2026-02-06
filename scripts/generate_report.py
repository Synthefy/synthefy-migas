#!/usr/bin/env python3
"""
Generate a post-eval report: run bar plots and optionally LaTeX table, then write a Markdown report.

Usage:
  uv run python scripts/generate_report.py --results_dir ./results/suite/context_64
  uv run python scripts/generate_report.py --config scripts/post_eval_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure scripts and repo root are on path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def discover_stats_csv(results_dir: Path) -> Path | None:
    """Find stats_Context_*_allsamples.csv in results_dir."""
    if not results_dir.is_dir():
        return None
    candidates = list(results_dir.glob("stats_Context_*_allsamples.csv"))
    return sorted(candidates)[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate post-eval report (bar plots + optional LaTeX + Markdown)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing stats_Context_*_allsamples.csv and outputs/",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config with results_dir, datasets_dir, display names",
    )
    parser.add_argument(
        "--latex_config",
        type=str,
        default=None,
        help="Optional path to LaTeX generate_table config (scripts/latex/config_example.yaml)",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Do not run bar plots",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for report and plots (default: results_dir/report)",
    )
    args = parser.parse_args()

    results_dir = None
    if args.results_dir:
        results_dir = Path(args.results_dir)
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        if not results_dir and cfg.get("results_dir"):
            results_dir = Path(cfg["results_dir"]).resolve()
    if not results_dir or not results_dir.is_dir():
        print("Error: need --results_dir or --config with results_dir pointing to an existing directory.")
        return 1

    stats_csv = discover_stats_csv(results_dir)
    if not stats_csv:
        print(f"Error: no stats_Context_*_allsamples.csv found in {results_dir}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count datasets and models from CSV
    import pandas as pd
    df = pd.read_csv(stats_csv)
    n_datasets = len(df)
    model_cols = [c for c in df.columns if c.endswith("_mean_mae")]
    models = [c.replace("_mean_mae", "") for c in model_cols]

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

    # Run bar plots
    if not args.skip_plots:
        import subprocess
        cmd = [
            sys.executable,
            str(_script_dir / "plot_bars.py"),
            "--results_dir", str(results_dir),
            "--out_dir", str(out_dir),
        ]
        if args.config:
            cmd.extend(["--config", str(args.config)])
        r = subprocess.run(cmd)
        if r.returncode == 0:
            report_lines.append("### Bar plots")
            report_lines.append("")
            report_lines.append("- [bar_aggregate_mean_mae.pdf](bar_aggregate_mean_mae.pdf)")
            report_lines.append("- [bar_grouped_mean_mae.pdf](bar_grouped_mean_mae.pdf)")
            report_lines.append("- [bar_ttfm_win_pct.pdf](bar_ttfm_win_pct.pdf)")
            report_lines.append("- [bar_improvement_pct.pdf](bar_improvement_pct.pdf)")
            report_lines.append("- [bar_elo_mean_mae.pdf](bar_elo_mean_mae.pdf) (if multielo installed)")
            report_lines.append("")
        else:
            report_lines.append("Bar plots failed or skipped.")
            report_lines.append("")
    else:
        report_lines.append("Bar plots skipped (--skip_plots).")
        report_lines.append("")

    # Optional LaTeX table
    if args.latex_config and Path(args.latex_config).exists():
        import subprocess
        latex_dir = _script_dir / "latex"
        cmd = [
            sys.executable,
            str(latex_dir / "generate_table.py"),
            "--config", str(Path(args.latex_config).resolve()),
            "--output", str(out_dir / "results_table.tex"),
        ]
        r = subprocess.run(cmd, cwd=str(latex_dir))
        if r.returncode == 0:
            report_lines.append("### LaTeX table")
            report_lines.append("")
            report_lines.append("- [results_table.tex](results_table.tex)")
            report_lines.append("")
        else:
            report_lines.append("LaTeX table generation failed.")
            report_lines.append("")
    else:
        report_lines.append("To add a LaTeX table, run with `--latex_config scripts/latex/config_example.yaml` (and point csv_path in that config to this results dir).")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("Generated by `scripts/generate_report.py`.")
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
