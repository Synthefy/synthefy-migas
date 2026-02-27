#!/usr/bin/env python3
"""
Fetch Trading Economics news for a date range and convert to the format
expected by create_daily_summaries.py (date + llm_summary).

Fetches one calendar month at a time and processes in a stream (no giant list in memory).
Use --raw-output DIR to save each month as DIR/YYYY-MM.json; use --from-raw DIR (or a single file) to load from disk.

Two output modes:
- Default: one JSON with all news (same text for every dataset — not ideal).
- --by-symbol: group news by TE symbol and write one JSON per commodity stem
  (e.g. te_news_by_symbol/bl1_com.json). Uses --symbols-file (default: commodities_list.json).

API key: TRADING_ECONOMICS_API_KEY or TE_API_KEY. Rate limit: 1 req/s.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional

import requests


def sanitize_symbol_to_stem(symbol: str) -> str:
    """Same as te_fetch_historical: symbol -> filename stem (e.g. BL1:COM -> bl1_com)."""
    s = re.sub(r"[^\w]", "_", symbol.strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "commodity"


def ticker_from_symbol(sym: str) -> str:
    """Ticker part for matching: BL1:COM -> BL1, C 1:COM -> C 1."""
    s = str(sym).strip()
    if ":" in s:
        return s.split(":")[0].strip()
    return s


def build_ticker_to_stem(symbols: list[str]) -> dict:
    """Map TE-style ticker (or normalized) to our filename stem for each commodity."""
    out = {}
    for sym in symbols:
        stem = sanitize_symbol_to_stem(sym)
        ticker = ticker_from_symbol(sym)
        out[ticker] = stem
        # Also match normalized (lowercase, no spaces) for TE returning "C1" vs "C 1"
        key_norm = re.sub(r"\s+", "", ticker.lower())
        out[key_norm] = stem
    return out


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("TRADING_ECONOMICS_API_KEY") or os.environ.get("TE_API_KEY")
    if not key:
        raise SystemExit(
            "Error: Set TRADING_ECONOMICS_API_KEY or TE_API_KEY in the environment."
        )
    return key


def parse_date(item: dict) -> str:
    """Extract YYYY-MM-DD from TE news item date field."""
    val = item.get("date") or item.get("Date")
    if not val:
        return ""
    s = str(val).strip()
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


def _fetch_news_one_request(
    api_key: str, d1: str, d2: str, country: Optional[str] = None
) -> list:
    """Single request to TE news API for date range [d1, d2]. Country is normalized to lowercase for URL."""
    if country:
        # TE API accepts lowercase in path; normalize so "Commodity" and "commodity" both work
        country_normalized = country.strip().lower()
        country_esc = requests.utils.quote(country_normalized, safe="")
        url = f"https://api.tradingeconomics.com/news/country/{country_esc}"
    else:
        url = "https://api.tradingeconomics.com/news"
    params = {"c": api_key, "d1": d1, "d2": d2, "f": "json"}
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def _last_day_of_month(year: int, month: int):
    """Last day of the given calendar month."""
    if month == 12:
        return datetime(year + 1, 1, 1).date() - timedelta(days=1)
    return datetime(year, month + 1, 1).date() - timedelta(days=1)


def fetch_news_monthly(
    api_key: str,
    d1: str,
    d2: str,
    countries: Optional[List[str]] = None,
    *,
    sleep_seconds: float = 1.0,
    raw_output_dir: Optional[str] = None,
) -> Iterator[tuple[int, int, list]]:
    """
    Fetch news from TE one month at a time. Yields (year, month, page_items) for each month.
    If countries is None or empty, fetches all news; otherwise one request per country per month,
    results merged and deduped by id. When raw_output_dir is set, writes each month to
    raw_output_dir/YYYY-MM.json before yielding. Respects rate limit via sleep_seconds.
    """
    start = datetime.strptime(d1, "%Y-%m-%d").date()
    end = datetime.strptime(d2, "%Y-%m-%d").date()
    if start > end:
        return

    raw_dir = Path(raw_output_dir) if raw_output_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    # None or empty => fetch all (no country filter); otherwise list of countries
    country_list = [c.strip() for c in countries if c and str(c).strip()] if countries else []

    y, m = start.year, start.month
    request_count = 0
    while True:
        month_start = datetime(y, m, 1).date()
        month_end = _last_day_of_month(y, m)
        chunk_start = max(month_start, start)
        chunk_end = min(month_end, end)
        if chunk_start > end:
            break

        c_d1 = chunk_start.strftime("%Y-%m-%d")
        c_d2 = chunk_end.strftime("%Y-%m-%d")

        if not country_list:
            if request_count > 0:
                time.sleep(sleep_seconds)
            page_items = _fetch_news_one_request(api_key, c_d1, c_d2, None)
            request_count += 1
        else:
            # One request per country, merge and dedupe by id
            seen_ids: set = set()
            page_items = []
            for country in country_list:
                if request_count > 0:
                    time.sleep(sleep_seconds)
                items = _fetch_news_one_request(api_key, c_d1, c_d2, country)
                request_count += 1
                for item in items:
                    item_id = item.get("id") or item.get("Id")
                    if item_id is not None and item_id in seen_ids:
                        continue
                    if item_id is not None:
                        seen_ids.add(item_id)
                    page_items.append(item)

        if raw_dir is not None:
            month_file = raw_dir / f"{y}-{m:02d}.json"
            with open(month_file, "w", encoding="utf-8") as f:
                json.dump(page_items, f, indent=2, ensure_ascii=False)

        yield (y, m, page_items)

        if (y, m) == (end.year, end.month):
            break
        m += 1
        if m > 12:
            m = 1
            y += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch TE news for date range and output JSON for create_daily_summaries"
    )
    parser.add_argument(
        "--d1",
        type=str,
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--d2",
        type=str,
        required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Filter by a single country (e.g. 'united states', 'commodity'). Deprecated: prefer --countries.",
    )
    parser.add_argument(
        "--countries",
        type=str,
        nargs="*",
        default=None,
        help="Filter by one or more countries (e.g. --countries commodity or --countries commodity 'united states'). If not set, fetch all news.",
    )
    parser.add_argument(
        "--commodity-only",
        action="store_true",
        help="Keep only items where country or category is commodity-related",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/te_commodities/te_news_for_daily_summaries.json",
        help="Output JSON path (default: data/te_commodities/te_news_for_daily_summaries.json)",
    )
    parser.add_argument(
        "--by-symbol",
        action="store_true",
        help="Group news by symbol and write one JSON per commodity (under --output-dir-by-symbol)",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default="data/te_commodities/commodities_list.json",
        help="JSON: commodities_list.json (list of {Symbol, Name, ...}) or legacy {symbols: [...]}; required for --by-symbol",
    )
    parser.add_argument(
        "--output-dir-by-symbol",
        type=str,
        default="data/te_commodities/te_news_by_symbol",
        help="Directory for per-symbol news JSONs when using --by-symbol",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="Directory to save raw API response: one JSON per month (YYYY-MM.json) for reuse.",
    )
    parser.add_argument(
        "--from-raw",
        type=str,
        default=None,
        help="Load news from previously saved raw: directory of YYYY-MM.json files, or single JSON file (skip API fetch).",
    )
    args = parser.parse_args()

    # Resolve countries: --countries wins, else single --country, else None (fetch all)
    countries = args.countries
    if countries is not None and len(countries) == 0:
        countries = None
    if countries is None and args.country:
        countries = [args.country]

    def to_entry(item: dict) -> Optional[dict]:
        date_str = parse_date(item)
        if not date_str:
            return None
        title = (item.get("title") or item.get("Title") or "").strip()
        desc = (item.get("description") or item.get("Description") or "").strip()
        llm_summary = f"Title: {title}\n\n{desc}" if title else desc
        if not llm_summary.strip():
            return None
        return {"date": date_str, "llm_summary": llm_summary}

    seen_ids: set = set()
    out_list: list = []
    by_stem: dict[str, list] = defaultdict(list)
    our_symbols: list = []
    ticker_to_stem: dict = {}

    if args.by_symbol:
        path = Path(args.symbols_file)
        if not path.exists():
            raise SystemExit(f"--by-symbol: symbols file not found: {path}. Run te_fetch_commodities_list.py first.")
        with open(path, "r", encoding="utf-8") as f:
            sym_data = json.load(f)
        if isinstance(sym_data, list) and sym_data and isinstance(sym_data[0], dict) and "Symbol" in sym_data[0]:
            our_symbols = [item["Symbol"] for item in sym_data if item.get("Symbol")]
        elif isinstance(sym_data, dict) and "symbols" in sym_data:
            our_symbols = sym_data["symbols"]
        elif isinstance(sym_data, list):
            our_symbols = [s for s in sym_data if isinstance(s, str) and s.strip()]
        else:
            our_symbols = []
        if not our_symbols:
            raise SystemExit("--by-symbol: no symbols in --symbols-file")
        ticker_to_stem = build_ticker_to_stem(our_symbols)

    def process_batch(page_items: list) -> None:
        for item in page_items:
            if args.commodity_only:
                if (str(item.get("country", "")).lower() != "commodity") and (
                    str(item.get("category", "")).lower() != "commodity"
                ):
                    continue
            item_id = item.get("id") or item.get("Id")
            if item_id is not None and item_id in seen_ids:
                continue
            if item_id is not None:
                seen_ids.add(item_id)
            ent = to_entry(item)
            if not ent:
                continue
            if args.by_symbol:
                te_sym = (item.get("symbol") or item.get("Symbol") or "").strip()
                stem = None
                if te_sym:
                    ticker = ticker_from_symbol(te_sym)
                    stem = ticker_to_stem.get(ticker) or ticker_to_stem.get(
                        re.sub(r"\s+", "", ticker.lower())
                    )
                if stem is not None:
                    by_stem[stem].append(ent)
            else:
                out_list.append(ent)

    if args.from_raw:
        raw_path = Path(args.from_raw)
        if not raw_path.exists():
            raise SystemExit(f"Raw path not found: {raw_path}")
        print(f"Loading news from {raw_path} ...")
        if raw_path.is_dir():
            count = 0
            for f in sorted(raw_path.glob("*.json")):
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                page = data if isinstance(data, list) else []
                process_batch(page)
                count += len(page)
            print(f"Loaded {count} raw items from {len(list(raw_path.glob('*.json')))} monthly files.")
        else:
            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("items", data.get("news", []))
            if not isinstance(items, list):
                items = []
            process_batch(items)
            print(f"Loaded {len(items)} raw items.")
    else:
        api_key = get_api_key()
        countries_desc = f" countries={countries}" if countries else " (all news)"
        print(
            f"Fetching TE news from {args.d1} to {args.d2}{countries_desc} (raw to {args.raw_output or 'memory'}) ..."
        )
        total = 0
        for y, m, page_items in fetch_news_monthly(
            api_key,
            args.d1,
            args.d2,
            countries,
            sleep_seconds=1.0,
            raw_output_dir=args.raw_output,
        ):
            process_batch(page_items)
            total += len(page_items)
        print(f"Fetched {total} items.")
        if args.raw_output:
            print(f"Saved raw news by month to {args.raw_output} (--from-raw {args.raw_output}).")

    if args.commodity_only and not args.from_raw:
        print(f"Filtered to commodity-related (applied during stream).")

    if args.by_symbol:
        out_dir = Path(args.output_dir_by_symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, entries in sorted(by_stem.items()):
            p = out_dir / f"{stem}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            print(f"  {stem}: {len(entries)} items -> {p}")
        for sym in our_symbols:
            stem = sanitize_symbol_to_stem(sym)
            if stem not in by_stem:
                with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump([], f)
                print(f"  {stem}: 0 items (no matching news)")
        print(f"Next: run create_daily_summaries for each {out_dir}/*.json, then merge_all_te_text --summaries-dir te_daily_summaries")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out_list)} items to {out_path}")
    print(f"Next: create_daily_summaries --input {out_path} --output {out_path.parent / 'te_daily_summaries.json'}")


if __name__ == "__main__":
    main()
