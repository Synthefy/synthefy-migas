#!/usr/bin/env python3
"""
Fetch Media Cloud news by commodity name for a date range and write per-stem JSON
in the format expected by create_daily_summaries.py (date + llm_summary).

Uses commodity Name from commodities_list.json to build search queries. Outputs
one JSON per stem under --output-dir (e.g. mc_news_by_symbol/bl1_com.json) so
run_daily_summaries_by_symbol and merge_all_te_text can be used with MC text.

API key: MEDIACLOUD_API_KEY or MC_API_KEY. Sign up at https://search.mediacloud.org
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Optional: only used when fetching from Media Cloud
try:
    import mediacloud.api
except ImportError:
    mediacloud = None  # type: ignore


# Optional override: symbol or name -> Media Cloud query string (for odd TE names)
QUERY_OVERRIDES = {
    "WEGGS:COM": '"egg" price',
    "Eggs CH": '"egg" price',
    "S 1:COM": '"soybeans" price',
    "O 1:COM": '"oat" price',
    "C 1:COM": '"corn" price',
    "XPDUSD:CUR": '"palladium" price',
    "XPTUSD:CUR": '"platinum" price',
    "XAUUSD:CUR": '"gold" price',
    "XAGUSD:CUR": '"silver" price',
}


def sanitize_symbol_to_stem(symbol: str) -> str:
    """Same as te_fetch_historical / te_fetch_news_to_summaries: symbol -> filename stem."""
    s = re.sub(r"[^\w]", "_", symbol.strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "commodity"


def build_symbol_to_name(commodities_list_path: Path) -> dict[str, str]:
    """Load commodities_list.json and return Symbol -> Name."""
    with open(commodities_list_path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for item in data:
        sym = item.get("Symbol") or item.get("symbol")
        name = item.get("Name") or item.get("name")
        if sym and name:
            out[str(sym).strip()] = str(name).strip()
    return out


def name_to_query(name: str, symbol: str, query_suffix: str | None) -> str:
    """Build Media Cloud query from commodity name (and optional override)."""
    if symbol in QUERY_OVERRIDES:
        return QUERY_OVERRIDES[symbol]
    if name in QUERY_OVERRIDES:
        return QUERY_OVERRIDES[name]
    # Phrase in quotes to reduce noise; optional suffix e.g. " price"
    phrase = name.lower().strip()
    if query_suffix:
        return f'"{phrase}" {query_suffix}'
    return f'"{phrase}"'


def get_date_range_from_raw(raw_dir: Path, stem: str) -> tuple[date, date] | None:
    """Read raw/{stem}.csv and return (min date, max date) from column t."""
    csv_path = raw_dir / f"{stem}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, nrows=0)
    if "t" not in df.columns:
        df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path)
    if "t" not in df.columns or df.empty:
        return None
    df["t"] = pd.to_datetime(df["t"])
    min_ts = df["t"].min()
    max_ts = df["t"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        return None
    return (min_ts.date(), max_ts.date())


def _handle_fetch_error(e: Exception, msg: str, attempt: int, max_retries: int) -> None:
    """Log fetch error and optionally suggest --symbol-delay."""
    if attempt < max_retries:
        backoff = 10 * (attempt + 1)
        print(f"    {msg} Retry in {backoff}s (attempt {attempt + 1}/{max_retries + 1})...", file=sys.stderr)
        time.sleep(backoff)
    else:
        print(f"    Error: {msg} {e}. Try --symbol-delay 3 or higher.", file=sys.stderr)


def get_api_key() -> str:
    """Get Media Cloud API key from environment."""
    key = os.environ.get("MEDIACLOUD_API_KEY") or os.environ.get("MC_API_KEY")
    if not key:
        raise SystemExit(
            "Error: Set MEDIACLOUD_API_KEY or MC_API_KEY in the environment. "
            "Sign up at https://search.mediacloud.org"
        )
    return key


def fetch_stories_for_date_range(
    api: "mediacloud.api.SearchApi",
    query: str,
    start_date: date,
    end_date: date,
    collection_ids: list[int],
    language: str | None,
    page_size: int,
    delay_seconds: float,
) -> list[dict]:
    """Paginate story_list and return all stories. Each story has title, url, publish_date."""
    if language:
        query = f"{query} AND language:{language}"
    all_stories = []
    pagination_token = None
    while True:
        page, pagination_token = api.story_list(
            query,
            start_date=start_date,
            end_date=end_date,
            collection_ids=collection_ids,
            pagination_token=pagination_token,
            page_size=page_size,
        )
        all_stories.extend(page)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        if not pagination_token:
            break
    return all_stories


def story_to_entry(story: dict) -> dict:
    """Convert Media Cloud story to create_daily_summaries input entry."""
    pub = story.get("publish_date")
    if hasattr(pub, "strftime"):
        date_str = pub.strftime("%Y-%m-%d")
    elif isinstance(pub, str):
        date_str = pub[:10] if len(pub) >= 10 else pub
    else:
        return None
    title = (story.get("title") or "").strip()
    url = (story.get("url") or "").strip()
    llm_summary = f"Title: {title}\n\nSource: {url}" if title else f"Source: {url}"
    return {"date": date_str, "llm_summary": llm_summary, "url": url}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Media Cloud news by commodity name and write per-stem JSON for create_daily_summaries"
    )
    parser.add_argument(
        "--d1",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (if not set, use per-stem range from raw CSV)",
    )
    parser.add_argument(
        "--d2",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (if not set, use per-stem range from raw CSV)",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=Path("data/te_commodities/te_commodity_symbols.json"),
        help="JSON with 'symbols' list (same as te_fetch_historical)",
    )
    parser.add_argument(
        "--commodities-list",
        type=Path,
        default=Path("data/te_commodities/commodities_list.json"),
        help="JSON list of commodities with Symbol and Name (from te_fetch_commodities_list)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/te_commodities/raw"),
        help="Directory of raw CSVs; used to get date range per stem when --d1/--d2 not set",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_commodities/mc_news_by_symbol"),
        help="Directory to write per-stem news JSONs",
    )
    parser.add_argument(
        "--collection-ids",
        type=str,
        default="34412234",
        help="Comma-separated Media Cloud collection IDs (default: US national 34412234)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language filter, e.g. en (default: en); empty to disable",
    )
    parser.add_argument(
        "--query-suffix",
        type=str,
        default="price",
        help="Append to name in query, e.g. 'price' (default: price); empty for name only",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Media Cloud story_list page size (default: 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between pagination requests within one symbol (default: 0.5)",
    )
    parser.add_argument(
        "--symbol-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between symbols to avoid rate limits (default: 2.0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per symbol on API error (default: 2)",
    )
    args = parser.parse_args()

    if mediacloud is None:
        print("Error: mediacloud package not installed. Run: uv sync --extra mediacloud", file=sys.stderr)
        sys.exit(1)

    api_key = get_api_key()
    mc_search = mediacloud.api.SearchApi(api_key)

    collection_ids = [int(x.strip()) for x in args.collection_ids.split(",") if x.strip()]
    query_suffix = args.query_suffix.strip() or None
    language = args.language.strip() or None

    with open(args.symbols_file, encoding="utf-8") as f:
        sym_data = json.load(f)
    symbols = sym_data.get("symbols", []) if isinstance(sym_data, dict) else sym_data
    if not symbols:
        print("Error: no symbols in --symbols-file", file=sys.stderr)
        sys.exit(1)

    symbol_to_name = build_symbol_to_name(args.commodities_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_start = args.d1
    global_end = args.d2
    if global_start and global_end:
        try:
            start_d = date.fromisoformat(global_start)
            end_d = date.fromisoformat(global_end)
        except ValueError:
            print("Error: --d1 and --d2 must be YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
    else:
        start_d = end_d = None

    for i, symbol in enumerate(symbols):
        # Delay between symbols to avoid rate limits (skip before first symbol)
        if i > 0 and args.symbol_delay > 0:
            time.sleep(args.symbol_delay)

        stem = sanitize_symbol_to_stem(symbol)
        name = symbol_to_name.get(symbol) or symbol
        query = name_to_query(name, symbol, query_suffix)

        if start_d is not None and end_d is not None:
            range_start, range_end = start_d, end_d
        else:
            dr = get_date_range_from_raw(args.raw_dir, stem)
            if not dr:
                print(f"  Skip {stem}: no raw CSV or date range (use --d1/--d2)")
                out_path = args.output_dir / f"{stem}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
                continue
            range_start, range_end = dr

        print(f"  {stem} ({name}): query={query!r} {range_start} to {range_end}")
        stories = []
        for attempt in range(args.retries + 1):
            try:
                stories = fetch_stories_for_date_range(
                    mc_search,
                    query,
                    range_start,
                    range_end,
                    collection_ids,
                    language,
                    args.page_size,
                    args.delay,
                )
                break
            except json.JSONDecodeError as e:
                _handle_fetch_error(e, "API returned non-JSON or empty response (often rate limit).", attempt, args.retries)
            except Exception as e:
                if "Expecting value" in str(e) or "JSON" in str(e):
                    _handle_fetch_error(e, "API returned non-JSON or empty response (often rate limit).", attempt, args.retries)
                else:
                    if attempt < args.retries:
                        backoff = 10 * (attempt + 1)
                        print(f"    Error: {e}. Retry in {backoff}s (attempt {attempt + 1}/{args.retries + 1})...", file=sys.stderr)
                        time.sleep(backoff)
                    else:
                        print(f"    Error fetching: {e}", file=sys.stderr)

        entries = []
        for s in stories:
            ent = story_to_entry(s)
            if ent:
                entries.append(ent)

        out_path = args.output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"    -> {len(entries)} items -> {out_path}")

    print(
        "\nNext: run_daily_summaries_by_symbol --news-dir "
        f"{args.output_dir} --output-dir data/te_commodities/mc_daily_summaries"
    )
    print("Then: merge_all_te_text --summaries-dir data/te_commodities/mc_daily_summaries --data-dir data/te_commodities")


if __name__ == "__main__":
    main()
