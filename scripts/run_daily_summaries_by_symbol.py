#!/usr/bin/env python3
"""
Run create_daily_summaries for each per-symbol news JSON (from te_fetch_news_to_summaries --by-symbol).

Input: te_news_by_symbol/*.json
Output: te_daily_summaries/{stem}.json (one per stem). Requires LLM server.
Then run: merge_all_te_text --summaries-dir data/te_commodities/te_daily_summaries
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run create_daily_summaries for each te_news_by_symbol/*.json"
    )
    parser.add_argument(
        "--news-dir",
        type=str,
        default="data/te_commodities/te_news_by_symbol",
        help="Directory of per-symbol news JSONs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/te_commodities/te_daily_summaries",
        help="Directory to write per-symbol daily summaries JSONs",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8004/v1",
        help="LLM server URL for create_daily_summaries",
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
        help="Skip stems whose news JSON is empty (0 items)",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Process all dates (do not filter to last 2.5 years); passed to create_daily_summaries",
    )
    args = parser.parse_args()

    news_dir = Path(args.news_dir)
    output_dir = Path(args.output_dir)
    if not news_dir.is_dir():
        print(f"Error: news dir not found: {news_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    create_script = script_dir / "create_daily_summaries.py"
    repo_root = script_dir.parent

    json_files = sorted(news_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files in {news_dir}", file=sys.stderr)
        sys.exit(0)

    for jpath in json_files:
        stem = jpath.stem
        out_path = output_dir / f"{stem}.json"
        if args.skip_empty:
            import json
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                print(f"Skip {stem}: empty news")
                continue
        cmd = [
            sys.executable,
            str(create_script),
            "--input",
            str(jpath.resolve()),
            "--output",
            str(out_path.resolve()),
            "--llm-url",
            args.llm_url,
            "--llm-model",
            args.llm_model,
        ]
        if getattr(args, "full_range", False):
            cmd.append("--full-range")
        print(f"Daily summaries for {stem} ...")
        ret = subprocess.run(cmd, cwd=str(repo_root))
        if ret.returncode != 0:
            print(f"Warning: failed for {stem}", file=sys.stderr)

    print(f"\nDone. Next: merge_all_te_text --summaries-dir {output_dir}")


if __name__ == "__main__":
    main()
