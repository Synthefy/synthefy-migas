#!/usr/bin/env python3
"""
Enrich electricity MC news by fetching each story URL and extracting article text.

Calls the shared mc_enrich_news_from_urls logic with electricity paths:
  input:  data/te_electricity/mc_news/
  output: data/te_electricity/mc_news_enriched/

For each item with a url, fetches the page and extracts main text with trafilatura,
then replaces llm_summary with "Title: ...\n\n" + extracted body (truncated to --max-chars).
Use the enriched dir as --mc-dir when running merge_mc_into_te_gaps.py for more text.

Requires: trafilatura (uv sync --extra mediacloud).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich electricity MC news JSONs by fetching URLs and extracting article text"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/te_electricity/mc_news"),
        help="Directory of per-stem MC news JSONs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_electricity/mc_news_enriched"),
        help="Directory to write enriched JSONs",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between URL fetches (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Request timeout per URL in seconds (default: 15)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Max characters of extracted text per article (default: 4000)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Process only this stem (e.g. gbrelepri_com); default: all JSONs in input-dir",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    enrich_script = repo_root / "scripts" / "mc_enrich_news_from_urls.py"
    if not enrich_script.exists():
        print(f"Error: {enrich_script} not found", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(enrich_script),
        "--input-dir",
        str(args.input_dir.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--delay",
        str(args.delay),
        "--timeout",
        str(args.timeout),
        "--max-chars",
        str(args.max_chars),
    ]
    if args.stem:
        cmd.extend(["--stem", args.stem])

    rc = subprocess.run(cmd, cwd=str(repo_root))
    if rc.returncode != 0:
        sys.exit(rc.returncode)
    print(
        "\nNext: uv run python scripts/te_electricity/merge_mc_into_te_gaps.py "
        "--mc-dir data/te_electricity/mc_news_enriched"
    )


if __name__ == "__main__":
    main()
