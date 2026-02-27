#!/usr/bin/env python3
"""
Build per-(country, symbol) news JSON in the format expected by create_daily_summaries.py.

Reads raw news from te_news_raw_old and manifest (or exploration_ranked.csv); for each stem
writes news_for_summaries/{stem}.json with a list of { "date", "llm_summary", "title", "url" }
(one entry per article). Use raw title+description as llm_summary so create_daily_summaries
can group by date and produce one vLLM summary per day.

Next: run run_daily_summaries_te_countries.py (or run_daily_summaries_by_symbol with these dirs),
then build_country_indicator_text.py --summaries-dir data/te_countries/te_daily_summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_country_config(path: Path) -> tuple[list[str], dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    countries = data.get("countries", [])
    display_names = [c["display_name"] for c in countries]
    alias_to_display = {}
    for c in countries:
        display = c["display_name"]
        for alias in c.get("aliases", []) + [display]:
            key = str(alias).strip().lower()
            alias_to_display[key] = display
    return display_names, alias_to_display


def parse_news_date(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        dt = datetime.strptime(s.split(" ")[0], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def load_raw_news(
    raw_dir: Path,
    alias_to_display: dict[str, str],
    allowed: set[str],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    out: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for fp in sorted(raw_dir.glob("*.json")):
        stem = fp.stem
        if len(stem) != 7 or stem[4] != "-":
            continue
        try:
            y, m = stem.split("-")
            if len(y) != 4 or len(m) != 2 or not y.isdigit() or not m.isdigit():
                continue
        except Exception:
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else []):
            country_raw = (item.get("country") or item.get("Country") or "").strip()
            symbol = (item.get("symbol") or item.get("Symbol") or "").strip()
            if not symbol:
                continue
            display = alias_to_display.get(country_raw.lower().strip())
            if display is None or display not in allowed:
                continue
            date_str = parse_news_date(item.get("date") or item.get("Date"))
            if not date_str:
                continue
            title = (item.get("title") or item.get("Title") or "").strip()
            desc = (item.get("description") or item.get("Description") or "").strip()
            text = f"{title}\n\n{desc}".strip() if title else desc
            if not text:
                continue
            out[(display, symbol)].append((date_str, text))
    return dict(out)


def sanitize_stem(country: str, symbol: str) -> str:
    c = re.sub(r"[^\w\s]", "", country).strip().replace(" ", "_")
    s = re.sub(r"[^\w]", "_", symbol.strip())
    return f"{c}_{s}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build news JSON for vLLM daily summarization (create_daily_summaries input)",
    )
    parser.add_argument(
        "--raw-news-dir",
        type=Path,
        default=Path("data/te_commodities/te_news_raw_old"),
        help="Directory of monthly YYYY-MM.json raw news",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/te_countries/raw/manifest.json"),
        help="manifest.json from fetch_country_indicator_historical (country, symbol, stem)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries/news_for_summaries"),
        help="Directory for per-stem JSONs (create_daily_summaries input format)",
    )
    parser.add_argument(
        "--countries-json",
        type=Path,
        default=Path("data/te_countries/te_countries.json"),
        help="Country config for normalizing names",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Process only this stem; default: all in manifest",
    )
    args = parser.parse_args()

    if not args.raw_news_dir.is_dir():
        raise SystemExit(f"Raw news dir not found: {args.raw_news_dir}")
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}. Run fetch_country_indicator_historical first.")
    if not args.countries_json.exists():
        raise SystemExit(f"Countries JSON not found: {args.countries_json}")

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    display_names, alias_to_display = load_country_config(args.countries_json)
    allowed = set(display_names)

    print("Loading raw news ...")
    pairs = load_raw_news(args.raw_news_dir, alias_to_display, allowed)
    print(f"  Loaded {len(pairs)} (country, symbol) pairs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        stem = entry.get("stem", "")
        country = entry.get("country", "")
        symbol = entry.get("symbol", "")
        if args.stem and stem != args.stem:
            continue
        key = (country, symbol)
        entries_list = pairs.get(key, [])
        if not entries_list:
            print(f"  Skip {stem}: no news")
            continue
        # create_daily_summaries expects list of { date, llm_summary, title?, url? }
        out_list = []
        for date_str, text in entries_list:
            title = (text.split("\n\n")[0][:300] if text else "") or ""
            out_list.append({
                "date": date_str,
                "llm_summary": text,
                "title": title,
                "url": "",
            })
        out_path = args.output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_list, f, indent=2, ensure_ascii=False)
        print(f"  {stem}: {len(out_list)} articles -> {out_path}")

    print("Done. Next: run vLLM daily summarization, then merge with --summaries-dir.")


if __name__ == "__main__":
    main()
