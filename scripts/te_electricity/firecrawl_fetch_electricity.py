#!/usr/bin/env python3
"""
Fetch on-topic news for TE electricity series using Firecrawl Search + Scrape.

Unlike Media Cloud (titles + URLs only), Firecrawl returns full page markdown so
each item has substantive content for the daily summarizer. That yields useful
TTFM metadata: state description and forward-looking signals.

Reads data/te_electricity/electricity_queries.json. For each stem, computes gap
dates (raw CSV dates with no TE news), then runs Firecrawl search with a focused
query + date filter and scrapes each result for markdown. Writes per-stem JSON
in the same format as MC: list of { date, llm_summary, url } with llm_summary =
"Title: ...\n\n" + article markdown (truncated), so create_daily_summaries gets
real content.

Requires: firecrawl-py and FIRECRAWL_API_KEY.
Usage:
  uv run python scripts/te_electricity/firecrawl_fetch_electricity.py
  uv run python scripts/te_electricity/firecrawl_fetch_electricity.py --stem espelepri_com --d1 2024-01-01 --d2 2024-12-31

Then use output as --mc-dir in merge_mc_into_te_gaps.py, then run_daily_summaries_by_symbol and merge_all_te_text.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None  # type: ignore

# Max chars per article in llm_summary (daily summarizer truncates to 600 per article anyway)
MAX_MARKDOWN_CHARS = 8000
# Min body length to keep an entry (skip title+URL only; model needs substantive text for forecast)
MIN_BODY_CHARS = 150


def _mc_query_to_plain_query(mc_query: str) -> str:
    """Turn Media Cloud-style query into a plain search string for Firecrawl."""
    if not mc_query:
        return ""
    # Remove AND/OR and quoted phrases, keep meaningful tokens
    s = re.sub(r'\s+AND\s+', " ", mc_query, flags=re.IGNORECASE)
    s = re.sub(r'\s+OR\s+', " ", s, flags=re.IGNORECASE)
    s = re.sub(r'"([^"]*)"', r"\1", s)
    s = re.sub(r"\s*\(\s*", " ", s)
    s = re.sub(r"\s*\)\s*", " ", s)
    return " ".join(s.split())


def get_api_key() -> str:
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        raise SystemExit(
            "Error: Set FIRECRAWL_API_KEY. Get one at https://firecrawl.dev"
        )
    return key


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


def _scrape_url(app: "FirecrawlApp", url: str, delay: float) -> str:
    """Fetch URL with Firecrawl scrape; return markdown or empty string."""
    try:
        out = app.scrape_url(url, params={"formats": ["markdown"]})
        if isinstance(out, dict):
            md = out.get("markdown") or out.get("data", {}).get("markdown") or ""
        else:
            md = getattr(out, "markdown", None) or getattr(getattr(out, "data", None), "markdown", None) or ""
        return (md or "")[:MAX_MARKDOWN_CHARS]
    except Exception:
        return ""
    finally:
        if delay > 0:
            time.sleep(delay)


def _debug_search_response(result, query: str) -> None:
    """Print Firecrawl search response structure to stderr for debugging."""
    import pprint
    print("[firecrawl debug] query:", query[:80], file=sys.stderr)
    print("[firecrawl debug] result type:", type(result).__name__, file=sys.stderr)
    if isinstance(result, dict):
        print("[firecrawl debug] result keys:", list(result.keys()), file=sys.stderr)
        raw = result.get("data")
        print("[firecrawl debug] result['data'] type:", type(raw).__name__ if raw is not None else None, file=sys.stderr)
        if isinstance(raw, dict):
            print("[firecrawl debug] result['data'] keys:", list(raw.keys()), file=sys.stderr)
            for k in ("web", "results"):
                arr = raw.get(k)
                if arr is not None:
                    print(f"[firecrawl debug] result['data']['{k}'] len:", len(arr), file=sys.stderr)
                    if arr:
                        first = arr[0]
                        print(f"[firecrawl debug] first item type: {type(first).__name__}", file=sys.stderr)
                        if isinstance(first, dict):
                            print("[firecrawl debug] first item keys:", list(first.keys()), file=sys.stderr)
                            print("[firecrawl debug] first item url:", first.get("url"), file=sys.stderr)
                        else:
                            print("[firecrawl debug] first item dir:", [x for x in dir(first) if not x.startswith("_")][:20], file=sys.stderr)
        elif isinstance(raw, list):
            print("[firecrawl debug] result['data'] (list) len:", len(raw), file=sys.stderr)
    else:
        print("[firecrawl debug] result.data: <no attr> (SDK may return SearchData with .web/.news)", file=sys.stderr)
        for attr in ("web", "news", "results"):
            val = getattr(result, attr, None)
            if val is not None:
                print(f"[firecrawl debug] result.{attr} len: {len(val) if isinstance(val, (list, tuple)) else '?'}", file=sys.stderr)
                if isinstance(val, (list, tuple)) and val:
                    first = val[0]
                    print(f"[firecrawl debug] result.{attr}[0] type: {type(first).__name__}", file=sys.stderr)
                    if isinstance(first, dict):
                        print(f"[firecrawl debug] result.{attr}[0] keys: {list(first.keys())}", file=sys.stderr)
                    break


def search_and_scrape(
    app: "FirecrawlApp",
    query: str,
    day: date,
    limit: int = 10,
    delay_seconds: float = 1.0,
    debug_first: bool = False,
    min_body_chars: int = MIN_BODY_CHARS,
) -> list[dict]:
    """Run Firecrawl search for one day; scrape each result for markdown. Return list of { date, llm_summary, url }."""
    try:
        from firecrawl import ScrapeOptions
        scrape_opts = ScrapeOptions(formats=["markdown"])
    except (ImportError, AttributeError):
        scrape_opts = None

    kwargs = {"limit": limit}
    if scrape_opts is not None:
        kwargs["scrape_options"] = scrape_opts
    try:
        result = app.search(query, **kwargs)
    except TypeError:
        result = app.search(query, limit=limit)
    except Exception as e:
        if debug_first:
            print(f"[firecrawl debug] search() raised: {e}", file=sys.stderr)
        raise

    if debug_first:
        _debug_search_response(result, query)

    date_str = day.strftime("%Y-%m-%d")
    entries = []
    # Firecrawl API returns { "data": { "web": [...], "news": [...] } }. SDK returns SearchData (no .data attr) with .web, .news
    raw = getattr(result, "data", None) if not isinstance(result, dict) else result.get("data")
    if raw is None:
        # SDK returns the data payload as result (SearchData): has .web and optionally .news
        raw = result
    if isinstance(raw, dict):
        data = raw.get("web") or raw.get("news") or raw.get("results") or []
    elif hasattr(raw, "web") or hasattr(raw, "news"):
        data = list(getattr(raw, "web", None) or []) + list(getattr(raw, "news", None) or [])
    elif isinstance(raw, list):
        data = raw
    elif isinstance(result, list):
        data = result
    else:
        data = []
    if not isinstance(data, list):
        data = []

    def _get(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    for item in data:
        url = (_get(item, "url") or _get(_get(item, "metadata"), "sourceURL") or "").strip()
        title = (_get(item, "title") or _get(_get(item, "metadata"), "title") or "").strip()
        meta = _get(item, "metadata")
        markdown = (
            _get(item, "markdown")
            or (_get(meta, "markdown") if meta is not None else None)
            or _get(item, "content")
            or _get(item, "snippet")  # news results often have snippet
            or ""
        )
        if isinstance(markdown, str) and len(markdown) > MAX_MARKDOWN_CHARS:
            markdown = markdown[:MAX_MARKDOWN_CHARS] + "..."
        body = (markdown or "").strip()
        # If search didn't return body, scrape URL so we have substantive content for TTFM
        if not body and url:
            body = _scrape_url(app, url, delay_seconds)
        if not url:
            continue
        # Only keep entries with substantive body text (skip title+URL only; useless for TTFM forecast)
        if min_body_chars > 0 and len(body) < min_body_chars:
            continue
        llm_summary = f"Title: {title}\n\n{body}" if body else f"Title: {title}\n\nSource: {url}"
        entries.append({"date": date_str, "llm_summary": llm_summary, "url": url})

    if delay_seconds > 0 and data:
        time.sleep(delay_seconds)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch electricity news via Firecrawl (search + scrape) for gap dates"
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
        default=Path("data/te_electricity/fc_news"),
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
        "--limit",
        type=int,
        default=8,
        help="Max search results per day (default 8)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between Firecrawl requests (default 2)",
    )
    parser.add_argument(
        "--no-gaps",
        action="store_true",
        help="Fetch for all raw dates (ignore TE coverage); else only gap dates",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print Firecrawl API response structure for first request (debug 0 items)",
    )
    parser.add_argument(
        "--min-body-chars",
        type=int,
        default=MIN_BODY_CHARS,
        metavar="N",
        help=f"Keep only results with at least N chars of body text (default {MIN_BODY_CHARS}); 0 = keep all",
    )
    args = parser.parse_args()

    if FirecrawlApp is None:
        print(
            "Error: firecrawl-py not installed. Run: uv add firecrawl-py",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    api_key = get_api_key()
    app = FirecrawlApp(api_key=api_key)
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
        if args.d1 or args.d2:
            if args.d1:
                sorted_dates = [d for d in sorted_dates if d >= args.d1]
            if args.d2:
                sorted_dates = [d for d in sorted_dates if d <= args.d2]
        if not sorted_dates:
            print(f"  {stem}: no dates in range")
            continue

        entry = config[stem]
        name = entry.get("name", stem)
        # Prefer dedicated Firecrawl query (tighter, on-topic); MC query tended to go off-topic
        base_query = entry.get("firecrawl_query") or _mc_query_to_plain_query(entry.get("query", "")) or f"{name} wholesale electricity spot price"
        if not entry.get("firecrawl_query"):
            base_query = f"{base_query} electricity price news"

        all_entries = []
        for i, d_str in enumerate(sorted_dates):
            day = date.fromisoformat(d_str)
            # Add date to query to bias results toward that day
            query = f"{base_query} {day.strftime('%B %d %Y')}"
            try:
                entries = search_and_scrape(
                    app,
                    query,
                    day,
                    limit=args.limit,
                    delay_seconds=args.delay,
                    debug_first=args.verbose and (i == 0),
                    min_body_chars=args.min_body_chars,
                )
                all_entries.extend(entries)
            except Exception as e:
                print(f"    {d_str}: {e}", file=sys.stderr)
            if (i + 1) % 50 == 0:
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
