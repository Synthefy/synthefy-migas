#!/usr/bin/env python3
"""
Merge Media Cloud news into TE news by symbol, filling only gap dates.

For each of the top-14 stems: load te_news_by_symbol/<stem>.json (TE text) and
raw/<stem>.csv (numeric date range). Gap dates = dates in raw that have no TE text.
Load MC per-stem JSON (from mc_fetch_top14 + mc_enrich_news_from_urls), add MC
entries only for gap dates, merge with TE (TE first; MC fills gaps), sort by date.
Write to --output-dir so downstream (create_daily_summaries, merge_text_numerical)
can use the filled series.

Usage:
  uv run python scripts/merge_mc_into_te_gaps.py \\
    --mc-dir data/te_commodities/mc_news_top14/enriched \\
    --te-dir data/te_commodities/te_news_by_symbol \\
    --raw-dir data/te_commodities/raw \\
    --output-dir data/te_commodities/te_news_by_symbol_with_mc_gaps
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_top14_stems(config_path: Path) -> list[str]:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return sorted(config.keys())


def get_te_dates(te_path: Path) -> set[str]:
    """Return set of YYYY-MM-DD from TE news JSON."""
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


def get_raw_dates(raw_path: Path) -> set[str]:
    """Return set of YYYY-MM-DD from raw CSV column t."""
    if not raw_path.exists():
        return set()
    df = pd.read_csv(raw_path)
    if "t" not in df.columns or df.empty:
        return set()
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    return set(df["t"].dt.strftime("%Y-%m-%d").astype(str))


def load_te_entries(te_path: Path) -> list[dict]:
    if not te_path.exists():
        return []
    with open(te_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_mc_entries(mc_path: Path) -> list[dict]:
    if not mc_path.exists():
        return []
    with open(mc_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge MC news into TE news by symbol for gap dates only"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/te_commodities/top14_queries.json"),
        help="Top-14 config (used for stem list)",
    )
    parser.add_argument(
        "--te-dir",
        type=Path,
        default=Path("data/te_commodities/te_news_by_symbol"),
        help="Directory of TE per-stem JSONs",
    )
    parser.add_argument(
        "--mc-dir",
        type=Path,
        default=Path("data/te_commodities/mc_news_top14/enriched"),
        help="Directory of MC per-stem JSONs (enriched preferred)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/te_commodities/raw"),
        help="Directory of raw CSVs (defines numeric date range per stem)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_commodities/te_news_by_symbol_with_mc_gaps"),
        help="Directory to write merged per-stem JSONs",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    stems = load_top14_stems(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        te_path = args.te_dir / f"{stem}.json"
        mc_path = args.mc_dir / f"{stem}.json"
        raw_path = args.raw_dir / f"{stem}.csv"

        te_entries = load_te_entries(te_path)
        te_dates = get_te_dates(te_path)
        raw_dates = get_raw_dates(raw_path)
        gap_dates = raw_dates - te_dates

        mc_entries = load_mc_entries(mc_path)
        mc_for_gaps = []
        for e in mc_entries:
            d = (e.get("date") or "")[:10]
            if d in gap_dates:
                mc_for_gaps.append(e)

        merged = te_entries + mc_for_gaps
        merged.sort(key=lambda x: (x.get("date") or ""))

        out_path = args.output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        n_te = len(te_entries)
        n_filled = len(mc_for_gaps)
        print(f"  {stem}: TE={n_te}, gaps filled={n_filled}, total={len(merged)} -> {out_path}")

    print(f"\nMerged output in {args.output_dir}. Use this as --news-dir for run_daily_summaries_by_symbol, then merge_all_te_text with same data-dir.")


if __name__ == "__main__":
    main()
