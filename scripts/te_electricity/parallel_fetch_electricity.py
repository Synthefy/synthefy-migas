#!/usr/bin/env python3
"""
Fetch on-topic news for TE electricity series using Parallel Web Search API.

Parallel returns LLM-optimized excerpts (up to 10k chars per result), so each
item has substantive content for the daily summarizer—state and forward-looking
info for TTFM, unlike Media Cloud's titles + URLs only.

Reads data/te_electricity/electricity_queries.json. For each stem, computes gap
dates (raw CSV dates with no TE news), then runs Parallel search with an
objective + search queries. Uses publish_date from results and builds
llm_summary from title + excerpts. Writes per-stem JSON in the same format as
MC: list of { date, llm_summary, url } for merge_mc_into_te_gaps → daily
summaries → merge_all_te_text.

Requires: parallel-web and PARALLEL_API_KEY.
Usage:
  uv run python scripts/te_electricity/parallel_fetch_electricity.py
  uv run python scripts/te_electricity/parallel_fetch_electricity.py --stem espelepri_com --d1 2024-01-01

Then use output as --mc-dir in merge_mc_into_te_gaps.py, then run_daily_summaries_by_symbol and merge_all_te_text.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from parallel import Parallel
except ImportError:
    Parallel = None  # type: ignore

MAX_EXCERPT_CHARS = 8000


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def get_raw_dates(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    df = pd.read_csv(raw_path)
    if "t" not in df.columns or df.empty:
        return set()
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    return set(df["t"].dt.strftime("%Y-%m-%d").astype(str))


def get_te_dates(te_path: Path) -> set[str]:
    if not te_path.exists():
        return set()
    with open(te_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return set()
    out = set()
    for item in data:
        d = item.get("date") or item.get("Date")
        if d and len(d) >= 10:
            out.add(str(d)[:10])
    return out


def search_parallel(
    client: "Parallel",
    name: str,
    d_str: str,
    objective_override: str | None = None,
    max_results: int = 10,
    excerpt_chars: int = 6000,
    delay_seconds: float = 1.0,
) -> list[dict]:
    """Run Parallel search for one day; return list of { date, llm_summary, url }."""
    objective = objective_override or (
        f"News and analysis about {name} wholesale electricity prices, "
        "day-ahead or spot market, and forward-looking forecasts or expectations."
    )
    search_queries = [
        f"{name} wholesale electricity price day-ahead spot {d_str}",
        f"{name} electricity spot market prices news {d_str}",
    ]

    try:
        search = client.beta.search(
            objective=objective,
            search_queries=search_queries,
            max_results=max_results,
            excerpts={"max_chars_per_result": excerpt_chars},
        )
    except Exception as e:
        print(f"    Parallel search error for {d_str}: {e}", file=sys.stderr)
        return []

    results = getattr(search, "results", None) or []
    entries = []
    for r in results:
        url = (getattr(r, "url", None) or "").strip()
        title = (getattr(r, "title", None) or "").strip()
        pub = getattr(r, "publish_date", None)
        if pub and len(str(pub)) >= 10:
            date_str = str(pub)[:10]
        else:
            date_str = d_str
        excerpts_list = getattr(r, "excerpts", None) or []
        body = "\n\n".join(excerpts_list) if excerpts_list else ""
        if isinstance(body, str) and len(body) > MAX_EXCERPT_CHARS:
            body = body[:MAX_EXCERPT_CHARS] + "..."
        llm_summary = f"Title: {title}\n\n{body}" if body else f"Title: {title}\n\nSource: {url}"
        entries.append({"date": date_str, "llm_summary": llm_summary, "url": url})

    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch electricity news via Parallel Web Search for gap dates"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/te_electricity/electricity_queries.json"),
        help="JSON: stem -> {name, query}",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/te_electricity/raw"),
        help="Raw CSVs (date range and stems)",
    )
    parser.add_argument(
        "--te-dir",
        type=Path,
        default=Path("data/te_commodities/te_news_by_symbol"),
        help="TE news JSONs (to compute gap dates)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_electricity/parallel_news"),
        help="Output per-stem JSONs",
    )
    parser.add_argument(
        "--d1",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: from raw CSV)",
    )
    parser.add_argument(
        "--d2",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: from raw CSV)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Process only this stem",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Max results per day (default 10)",
    )
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=6000,
        help="Max chars per result excerpt (default 6000)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between API requests (default 1.5)",
    )
    parser.add_argument(
        "--no-gaps",
        action="store_true",
        help="Fetch for all raw dates (ignore TE coverage); else only gap dates",
    )
    args = parser.parse_args()

    if Parallel is None:
        print(
            "Error: parallel-web not installed. Run: uv add parallel-web",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("PARALLEL_API_KEY")
    if not api_key:
        print(
            "Error: Set PARALLEL_API_KEY. Get one at https://platform.parallel.ai",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    client = Parallel(api_key=api_key)
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(config.keys())
    if args.stem:
        stems = [s for s in stems if s == args.stem]
    if not stems:
        print("No stems to process.", file=sys.stderr)
        sys.exit(0)

    for stem in stems:
        raw_path = args.raw_dir / f"{stem}.csv"
        te_path = args.te_dir / f"{stem}.json"
        raw_dates = get_raw_dates(raw_path)
        if not raw_dates:
            print(f"  Skip {stem}: no raw dates in {raw_path}")
            continue

        if args.no_gaps:
            target_dates = raw_dates
        else:
            te_dates = get_te_dates(te_path)
            target_dates = raw_dates - te_dates
        if not target_dates:
            print(f"  {stem}: no gap dates, skipping")
            continue

        sorted_dates = sorted(target_dates)
        if args.d1:
            sorted_dates = [d for d in sorted_dates if d >= args.d1]
        if args.d2:
            sorted_dates = [d for d in sorted_dates if d <= args.d2]
        if not sorted_dates:
            print(f"  {stem}: no dates in range")
            continue

        entry = config[stem]
        name = entry.get("name", stem)
        # Prefer dedicated objective (tighter, on-topic); MC-style query tended to go off-topic
        parallel_objective = entry.get("parallel_objective")

        all_entries = []
        for i, d_str in enumerate(sorted_dates):
            entries = search_parallel(
                client,
                name,
                d_str,
                objective_override=parallel_objective,
                max_results=args.max_results,
                excerpt_chars=args.excerpt_chars,
                delay_seconds=args.delay,
            )
            all_entries.extend(entries)
            if (i + 1) % 30 == 0:
                print(f"    ... {i + 1}/{len(sorted_dates)} dates")

        out_path = args.output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)
        print(f"  {stem}: {len(all_entries)} items -> {out_path}")

    print(
        "\nNext: merge_mc_into_te_gaps.py --mc-dir",
        str(args.output_dir),
    )
    print("Then: run_daily_summaries_by_symbol and merge_all_te_text (see scripts/te_electricity/README.md)")


if __name__ == "__main__":
    main()
