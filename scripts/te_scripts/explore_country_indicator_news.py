#!/usr/bin/env python3
"""
Explore Trading Economics raw news to find (country, symbol) pairs with the most text.

Loads monthly JSONs from te_news_raw_old, normalizes country to the 20 enterprise countries,
groups by (country, symbol), and outputs a ranked CSV. Optional: call TE forecast API
to intersect with indicators that have historical data and attach category names.

Usage:
  uv run python scripts/te_scripts/explore_country_indicator_news.py
  uv run python scripts/te_scripts/explore_country_indicator_news.py --raw-dir data/te_commodities/te_news_raw_old --countries-json data/te_countries/te_countries.json --output data/te_countries/exploration_ranked.csv
  uv run python scripts/te_scripts/explore_country_indicator_news.py --with-forecast  # requires TE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_country_config(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load te_countries.json. Returns (list of display_name, alias_lower -> display_name)."""
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
    """Extract YYYY-MM-DD from TE news item date field."""
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
    """
    Load all YYYY-MM.json from raw_dir. Filter to allowed countries, normalize country.
    Returns (country_display, symbol) -> list of (date_yyyy_mm_dd, title_plus_description).
    """
    out: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    files = sorted(raw_dir.glob("*.json"))
    for fp in files:
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
        items = data if isinstance(data, list) else []
        for item in items:
            country_raw = (item.get("country") or item.get("Country") or "").strip()
            symbol = (item.get("symbol") or item.get("Symbol") or "").strip()
            if not symbol:
                continue
            country_key = country_raw.lower().strip()
            display = alias_to_display.get(country_key)
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


def aggregate_stats(
    pairs: dict[tuple[str, str], list[tuple[str, str]]],
    from_date: str | None,
    to_date: str | None,
) -> list[dict]:
    """Aggregate per (country, symbol): news_count, unique_dates, date_min, date_max, months_with_news."""
    from_ts = pd.Timestamp(from_date) if from_date else None
    to_ts = pd.Timestamp(to_date) if to_date else None

    rows = []
    for (country, symbol), entries in pairs.items():
        dates = [e[0] for e in entries]
        if from_ts:
            dates = [d for d in dates if pd.Timestamp(d) >= from_ts]
        if to_ts:
            dates = [d for d in dates if pd.Timestamp(d) <= to_ts]
        if not dates:
            continue
        unique_dates = sorted(set(dates))
        date_min = min(unique_dates)
        date_max = max(unique_dates)
        # Calendar days between date_min and date_max (inclusive)
        days_in_range = (pd.Timestamp(date_max) - pd.Timestamp(date_min)).days + 1
        # Fraction of those days that have at least one news item
        daily_coverage_fraction = len(unique_dates) / days_in_range if days_in_range > 0 else None
        months = set(d[:7] for d in unique_dates)  # YYYY-MM
        total_months = None
        if from_date and to_date:
            r = pd.date_range(start=from_date[:7], end=to_date[:7], freq="ME")
            total_months = len(r)
        coverage = len(months) / total_months if total_months and total_months > 0 else None
        in_range_entries = [e for e in entries if (from_ts is None or pd.Timestamp(e[0]) >= from_ts) and (to_ts is None or pd.Timestamp(e[0]) <= to_ts)]
        rows.append({
            "country": country,
            "symbol": symbol,
            "news_count": len(in_range_entries),
            "unique_dates": len(unique_dates),
            "days_in_range": days_in_range,
            "daily_coverage_fraction": round(daily_coverage_fraction, 4) if daily_coverage_fraction is not None else None,
            "date_min": date_min,
            "date_max": date_max,
            "months_with_news": len(months),
            "coverage_fraction": round(coverage, 4) if coverage is not None else None,
        })
    return rows


def fetch_forecast_indicators(api_key: str, countries: list[str]) -> list[dict]:
    """Call TE forecast/country/{countries}, return list of indicator objects with HistoricalDataSymbol etc."""
    import requests
    # TE API accepts comma-separated countries in path
    slug_list = [c.lower().replace(" ", "%20") for c in countries]
    country_path = ",".join(slug_list)
    url = f"https://api.tradingeconomics.com/forecast/country/{country_path}"
    params = {"c": api_key, "f": "json"}
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def attach_forecast_metadata(
    rows: list[dict],
    forecast_indicators: list[dict],
) -> list[dict]:
    """Add category and HistoricalDataSymbol from forecast response. Match by country + symbol."""
    # Build (Country normalized, HistoricalDataSymbol) -> category
    symbol_to_category = {}
    for ind in forecast_indicators:
        country = (ind.get("Country") or ind.get("country") or "").strip()
        sym = (ind.get("HistoricalDataSymbol") or "").strip()
        cat = (ind.get("Category") or ind.get("category") or ind.get("Title") or "").strip()
        if country and sym:
            symbol_to_category[(country.lower(), sym)] = cat
    for r in rows:
        key = (r["country"].lower(), r["symbol"])
        r["category"] = symbol_to_category.get(key, "")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore TE raw news by (country, symbol) and output ranked CSV",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/te_commodities/te_news_raw_old"),
        help="Directory of monthly YYYY-MM.json raw news",
    )
    parser.add_argument(
        "--countries-json",
        type=Path,
        default=Path("data/te_countries/te_countries.json"),
        help="JSON with countries and aliases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/te_countries/exploration_ranked.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default="2015-01-01",
        help="Only count news on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default="2026-02-28",
        help="Only count news on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--with-forecast",
        action="store_true",
        help="Call TE forecast API to attach category and intersect with available indicators",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="After filtering, write at most N rows (default: all). If fewer than N pass filters, only those are written.",
    )
    parser.add_argument(
        "--min-days-in-range",
        type=int,
        default=365,
        help="Exclude (country, symbol) with fewer calendar days between date_min and date_max (default: 365). Use 0 to keep all.",
    )
    parser.add_argument(
        "--min-unique-dates",
        type=int,
        default=450,
        help="Exclude pairs with fewer unique days-with-news between their date_min and date_max (default: 450). Use 0 to keep all.",
    )
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        raise SystemExit(f"Raw dir not found: {args.raw_dir}")

    display_names, alias_to_display = load_country_config(args.countries_json)
    allowed = set(display_names)

    print(f"Loading raw news from {args.raw_dir} ...")
    pairs = load_raw_news(args.raw_dir, alias_to_display, allowed)
    print(f"  Found {len(pairs)} (country, symbol) pairs with news in the 20 countries.")

    rows = aggregate_stats(pairs, args.from_date, args.to_date)
    if not rows:
        print("No rows after aggregation.")
        return

    df = pd.DataFrame(rows)
    if args.min_days_in_range > 0:
        before = len(df)
        df = df[df["days_in_range"] >= args.min_days_in_range]
        print(f"  Kept {len(df)} pairs with days_in_range >= {args.min_days_in_range} (dropped {before - len(df)}).")
    if args.min_unique_dates > 0:
        before = len(df)
        df = df[df["unique_dates"] >= args.min_unique_dates]
        print(f"  Kept {len(df)} pairs with unique_dates >= {args.min_unique_dates} (dropped {before - len(df)}).")
    # Sort by daily news presence: first by daily_coverage_fraction (highest share of days with news),
    # then by unique_dates (so more days with news wins when fraction ties, e.g. 1.0 with 2530 days before 1.0 with 1 day)
    df = df.sort_values(
        ["daily_coverage_fraction", "unique_dates"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if args.with_forecast:
        api_key = os.environ.get("TRADING_ECONOMICS_API_KEY") or os.environ.get("TE_API_KEY")
        if not api_key:
            print("Warning: TRADING_ECONOMICS_API_KEY / TE_API_KEY not set; skipping forecast.")
        else:
            print("Fetching forecast indicators for 20 countries ...")
            try:
                forecast = fetch_forecast_indicators(api_key, display_names)
                print(f"  Got {len(forecast)} indicator entries.")
                rows = df.to_dict("records")
                rows = attach_forecast_metadata(rows, forecast)
                df = pd.DataFrame(rows)
            except Exception as e:
                print(f"  Forecast API error: {e}")

    if args.top is not None:
        df = df.head(args.top)
        if len(df) < args.top:
            print(f"  Note: --top {args.top} requested but only {len(df)} rows passed filters.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
