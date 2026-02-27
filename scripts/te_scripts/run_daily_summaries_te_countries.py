#!/usr/bin/env python3
"""
Run create_daily_summaries for each country-indicator news JSON (from build_news_for_summaries).

Calls scripts/run_daily_summaries_by_symbol.py with:
  --news-dir data/te_countries/news_for_summaries
  --output-dir data/te_countries/te_daily_summaries

Requires vLLM (or compatible) server running. Then run:
  build_country_indicator_text.py --summaries-dir data/te_countries/te_daily_summaries
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vLLM daily summarization for TE country-indicator news",
    )
    parser.add_argument(
        "--news-dir",
        type=Path,
        default=Path("data/te_countries/news_for_summaries"),
        help="Input dir (output of build_news_for_summaries)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries/te_daily_summaries"),
        help="Output dir for daily summaries JSONs",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8004/v1",
        help="LLM server URL",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model name",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip stems with empty news JSON",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Process all dates in news (do not filter to last 2.5 years); matches news_for_summaries date range",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent  # .../scripts/te_scripts
    repo_root = script_dir.parent.parent  # .../synthefy-ttfm
    runner = repo_root / "scripts" / "run_daily_summaries_by_symbol.py"
    if not runner.exists():
        print(f"Error: {runner} not found", file=sys.stderr)
        sys.exit(1)
    if not args.news_dir.is_dir():
        print(f"Error: news dir not found: {args.news_dir}. Run build_news_for_summaries first.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(runner),
        "--news-dir", str(args.news_dir.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--llm-url", args.llm_url,
        "--llm-model", args.llm_model,
    ]
    if args.skip_empty:
        cmd.append("--skip-empty")
    if args.full_range:
        cmd.append("--full-range")
    ret = subprocess.run(cmd, cwd=str(repo_root))
    if ret.returncode != 0:
        sys.exit(ret.returncode)
    print(f"\nNext: uv run python scripts/te_scripts/build_country_indicator_text.py --summaries-dir {args.output_dir}")


if __name__ == "__main__":
    main()
