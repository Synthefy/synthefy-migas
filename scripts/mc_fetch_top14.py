#!/usr/bin/env python3
"""
Fetch Media Cloud news for the top-14 TE commodities using curated per-symbol queries.

Reads data/te_commodities/top14_queries.json (stem -> name, query), fetches MC stories
for --d1/--d2, optionally filters by title relevance, and writes per-stem JSON in the
format expected by create_daily_summaries (date, llm_summary, url). Then run
mc_enrich_news_from_urls on the output, then merge_mc_into_te_gaps to fill TE gaps.

Media Cloud rate limit: certain endpoints are limited to 2 requests per minute.
Defaults use --delay 35 and --symbol-delay 35 to stay under that. See:
https://www.mediacloud.org/documentation/faqs
Query syntax: AND/OR capitalized, exact phrases in double quotes.
https://www.mediacloud.org/documentation/query-guide

API key: MEDIACLOUD_API_KEY or MC_API_KEY.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

try:
    import mediacloud.api
except ImportError:
    mediacloud = None  # type: ignore

# Keywords that suggest commodity/market relevance in title (case-insensitive)
TITLE_RELEVANCE_KEYWORDS = (
    "futures", "price", "prices", "commodity", "commodities",
    "rally", "falls", "surges", "barrel", "ounce", "bushel",
    "NYMEX", "COMEX", "ICE", "CBOT", "LME", "WTI", "Brent",
    "crude", "oil", "gas", "gold", "silver", "copper", "wheat",
    "corn", "soy", "sugar", "coffee", "cocoa", "natural gas",
    "Henry Hub", "TTF", "RBOB", "gasoline", "heating oil",
)


def get_api_key() -> str:
    key = os.environ.get("MEDIACLOUD_API_KEY") or os.environ.get("MC_API_KEY")
    if not key:
        raise SystemExit(
            "Error: Set MEDIACLOUD_API_KEY or MC_API_KEY. "
            "Sign up at https://search.mediacloud.org"
        )
    return key


def load_top14_config(config_path: Path) -> dict:
    """Load top14_queries.json: stem -> {name, query}."""
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def date_range_by_years(start_date: date, end_date: date, chunk_years: int) -> list[tuple[date, date]]:
    """Return (start, end) date pairs covering [start_date, end_date] in chunk_years-year blocks."""
    if chunk_years < 1:
        return [(start_date, end_date)]
    chunks = []
    s = start_date
    while s <= end_date:
        end_year = s.year + chunk_years - 1
        e = min(date(end_year, 12, 31), end_date)
        chunks.append((s, e))
        s = e + timedelta(days=1)
    return chunks


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
    """Paginate story_list and return all stories (title, url, publish_date).
    Waits delay_seconds after each request to respect MC rate limit (2 req/min).
    """
    if language:
        query = f"{query} AND language:{language}"
    all_stories = []
    pagination_token = None
    while True:
        time.sleep(delay_seconds)  # Wait before each request to respect rate limit
        page, pagination_token = api.story_list(
            query,
            start_date=start_date,
            end_date=end_date,
            collection_ids=collection_ids,
            pagination_token=pagination_token,
            page_size=page_size,
        )
        all_stories.extend(page)
        if not pagination_token:
            break
    return all_stories


def story_to_entry(story: dict) -> dict | None:
    """Convert MC story to {date, llm_summary, url}. Returns None if no date."""
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


def title_looks_relevant(title: str, stem: str, name: str) -> bool:
    """True if title contains at least one relevance keyword or the commodity name."""
    if not title:
        return False
    lower = title.lower()
    # Check curated keywords
    for kw in TITLE_RELEVANCE_KEYWORDS:
        if kw.lower() in lower:
            return True
    # Check commodity name (e.g. "Cocoa", "Natural Gas")
    name_words = name.lower().split()
    for w in name_words:
        if len(w) >= 3 and w in lower:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Media Cloud news for top-14 commodities using curated queries"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/te_commodities/top14_queries.json"),
        help="JSON: stem -> {name, query}",
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
        "--output-dir",
        type=Path,
        default=Path("data/te_commodities/mc_news_top14"),
        help="Directory to write per-stem JSONs",
    )
    parser.add_argument(
        "--collection-ids",
        type=str,
        default="34412234",
        help="Comma-separated Media Cloud collection IDs (default: US national)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language filter (default: en); empty to disable",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="MC story_list page size",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=35.0,
        help="Seconds to wait before each API request (MC rate limit: 2 req/min; default 35)",
    )
    parser.add_argument(
        "--symbol-delay",
        type=float,
        default=35.0,
        help="Seconds to wait before first request of each symbol (default 35)",
    )
    parser.add_argument(
        "--no-relevance-filter",
        action="store_true",
        help="Do not filter stories by title relevance",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per symbol on API error",
    )
    parser.add_argument(
        "--chunk-years",
        type=int,
        default=0,
        metavar="N",
        help="Fetch in N-year date chunks (default 0 = whole range). Use 1 to reduce pages per request.",
    )
    args = parser.parse_args()

    if mediacloud is None:
        print("Error: mediacloud not installed. Run: uv sync --extra mediacloud", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        start_d = date.fromisoformat(args.d1)
        end_d = date.fromisoformat(args.d2)
    except ValueError:
        print("Error: --d1 and --d2 must be YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    config = load_top14_config(args.config)
    collection_ids = [int(x.strip()) for x in args.collection_ids.split(",") if x.strip()]
    language = args.language.strip() or None
    api_key = get_api_key()
    api = mediacloud.api.SearchApi(api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(config.keys())
    for i, stem in enumerate(stems):
        if i > 0 and args.symbol_delay > 0:
            time.sleep(args.symbol_delay)

        entry = config[stem]
        name = entry.get("name", stem)
        query = entry.get("query", "")
        if not query:
            print(f"  Skip {stem}: no query in config")
            continue

        print(f"  {stem} ({name}): query={query[:60]}...")
        chunks = date_range_by_years(start_d, end_d, args.chunk_years or 0)
        stories = []
        for attempt in range(args.retries + 1):
            try:
                stories = []
                for c_start, c_end in chunks:
                    part = fetch_stories_for_date_range(
                        api,
                        query,
                        c_start,
                        c_end,
                        collection_ids,
                        language,
                        args.page_size,
                        args.delay,
                    )
                    stories.extend(part)
                break
            except json.JSONDecodeError as e:
                if attempt < args.retries:
                    backoff = 30 * (attempt + 1)
                    print(
                        f"    API returned non-JSON (rate limit or error). Retry in {backoff}s "
                        f"(attempt {attempt + 1}/{args.retries + 1})...",
                        file=sys.stderr,
                    )
                    time.sleep(backoff)
                else:
                    print(
                        f"    Error: {e}. Media Cloud may be rate-limiting (2 req/min). "
                        "Try --delay 45 --symbol-delay 45.",
                        file=sys.stderr,
                    )
            except Exception as e:
                if attempt < args.retries:
                    backoff = 30 * (attempt + 1)
                    print(f"    Retry in {backoff}s (attempt {attempt + 1}/{args.retries + 1})...", file=sys.stderr)
                    time.sleep(backoff)
                else:
                    print(f"    Error: {e}", file=sys.stderr)

        entries = []
        for s in stories:
            ent = story_to_entry(s)
            if not ent:
                continue
            if not args.no_relevance_filter:
                title = (s.get("title") or "").strip()
                if not title_looks_relevant(title, stem, name):
                    continue
            entries.append(ent)

        out_path = args.output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"    -> {len(entries)} items -> {out_path}")

    print("\nNext: mc_enrich_news_from_urls --input-dir", str(args.output_dir), "--output-dir", str(args.output_dir / "enriched"))
    print("Then: merge_mc_into_te_gaps --mc-dir", str(args.output_dir / "enriched"), "--te-dir data/te_commodities/te_news_by_symbol")


if __name__ == "__main__":
    main()
