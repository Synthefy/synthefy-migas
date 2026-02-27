#!/usr/bin/env python3
"""
Build per-(country, symbol) CSVs with daily text presence: t, text_present (bool).

Reads exploration_ranked.csv and raw news from te_news_raw_old; for each row writes
data/te_countries/daily_presence/{stem}.csv with one row per calendar day from
date_min to date_max and text_present = True if that day has ≥1 news item, else False.

Use these CSVs with plot_daily_presence.py to visually pick date ranges with consistent text.
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
    allowed_display_names: set[str],
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
            if display is None or display not in allowed_display_names:
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
    """Filename stem for (country, symbol), e.g. United_States_USURTOT."""
    c = re.sub(r"[^\w\s]", "", country).strip().replace(" ", "_")
    s = re.sub(r"[^\w]", "_", symbol.strip())
    return f"{c}_{s}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily presence CSVs (t, text_present) for each row in exploration_ranked",
    )
    parser.add_argument(
        "--ranked-csv",
        type=Path,
        default=Path("data/te_countries/exploration_ranked.csv"),
        help="Exploration ranked CSV with country, symbol, date_min, date_max",
    )
    parser.add_argument(
        "--raw-news-dir",
        type=Path,
        default=Path("data/te_commodities/te_news_raw_old"),
        help="Directory of monthly YYYY-MM.json raw news",
    )
    parser.add_argument(
        "--countries-json",
        type=Path,
        default=Path("data/te_countries/te_countries.json"),
        help="Country config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries/daily_presence"),
        help="Directory for daily presence CSVs",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Process only this stem (e.g. United_States_SPX); default: all in ranked CSV",
    )
    args = parser.parse_args()

    if not args.ranked_csv.exists():
        raise SystemExit(f"Ranked CSV not found: {args.ranked_csv}")
    if not args.raw_news_dir.is_dir():
        raise SystemExit(f"Raw news dir not found: {args.raw_news_dir}")

    display_names, alias_to_display = load_country_config(args.countries_json)
    allowed = set(display_names)

    print("Loading raw news ...")
    pairs = load_raw_news(args.raw_news_dir, alias_to_display, allowed)
    # (country, symbol) -> set of dates (YYYY-MM-DD) that have news
    dates_with_news: dict[tuple[str, str], set[str]] = {
        k: set(e[0] for e in v) for k, v in pairs.items()
    }
    print(f"  Loaded {len(dates_with_news)} (country, symbol) pairs.")

    df = pd.read_csv(args.ranked_csv)
    for _, row in df.iterrows():
        country = str(row["country"]).strip()
        symbol = str(row["symbol"]).strip()
        date_min = str(row["date_min"]).strip()
        date_max = str(row["date_max"]).strip()
        stem = sanitize_stem(country, symbol)
        if args.stem and stem != args.stem:
            continue
        key = (country, symbol)
        present_dates = dates_with_news.get(key, set())
        if not present_dates:
            print(f"  Skip {stem}: no news dates")
            continue
        dr = pd.date_range(start=date_min, end=date_max, freq="D")
        rows_out = [
            {"t": d.strftime("%Y-%m-%d"), "text_present": d.strftime("%Y-%m-%d") in present_dates}
            for d in dr
        ]
        out_df = pd.DataFrame(rows_out)
        out_path = args.output_dir / f"{stem}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        n_present = out_df["text_present"].sum()
        print(f"  {stem}: {len(out_df)} days, {int(n_present)} with text -> {out_path}")


if __name__ == "__main__":
    main()
