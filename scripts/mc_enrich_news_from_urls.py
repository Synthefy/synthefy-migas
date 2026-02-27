#!/usr/bin/env python3
"""
Enrich Media Cloud news JSONs by fetching each story URL and extracting article text.

Reads per-stem JSONs from mc_news_by_symbol (each item has date, llm_summary, url).
For each item with a url: fetches the page, extracts main text with trafilatura,
and replaces llm_summary with "Title: ...\n\n" + extracted body (truncated).
Writes to --output-dir (default: mc_news_by_symbol_enriched) so originals are unchanged.

Requires: uv sync --extra mediacloud (adds trafilatura).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path


def _title_from_llm_summary(llm_summary: str) -> str:
    """Get title from existing llm_summary (Title: ...\n\nSource: ...)."""
    if not llm_summary:
        return ""
    m = re.match(r"Title:\s*(.+?)(?:\n\n|$)", llm_summary, re.DOTALL)
    return m.group(1).strip() if m else llm_summary.split("\n")[0].strip()


def fetch_and_extract(url: str, timeout: int = 15, max_chars: int = 4000) -> str | None:
    """Fetch URL and extract main article text; return None on failure."""
    try:
        import requests
        import trafilatura
    except ImportError:
        return None
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TTFM-eval/1.0; +https://github.com/)"},
        )
        resp.raise_for_status()
        html = resp.text
        if not html:
            return None
        text = trafilatura.extract(html)
        if not text or not text.strip():
            return None
        text = text.strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich MC news JSONs by fetching URLs and extracting article text"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/te_commodities/mc_news_by_symbol"),
        help="Directory of per-stem news JSONs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_commodities/mc_news_by_symbol_enriched"),
        help="Directory to write enriched JSONs (default: .../mc_news_by_symbol_enriched)",
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
        help="Process only this stem (e.g. cl1_com); default: all JSONs in input-dir",
    )
    args = parser.parse_args()

    try:
        import trafilatura  # noqa: F401
    except ImportError:
        print("Error: trafilatura not installed. Run: uv sync --extra mediacloud", file=sys.stderr)
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(args.input_dir.glob("*.json"))
    if args.stem:
        json_files = [p for p in json_files if p.stem == args.stem]
    if not json_files:
        print("No JSON files to process.", file=sys.stderr)
        sys.exit(0)

    for jpath in json_files:
        stem = jpath.stem
        with open(jpath, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            items = [items]

        enriched = []
        ok = 0
        fail = 0
        for i, item in enumerate(items):
            url = (item.get("url") or "").strip()
            if not url:
                enriched.append(item)
                continue
            if i > 0:
                time.sleep(args.delay)
            title = _title_from_llm_summary(item.get("llm_summary") or "")
            body = fetch_and_extract(url, timeout=args.timeout, max_chars=args.max_chars)
            if body:
                new_summary = f"Title: {title}\n\n{body}" if title else body
                enriched.append({
                    "date": item.get("date", ""),
                    "llm_summary": new_summary,
                    "url": url,
                })
                ok += 1
            else:
                enriched.append(item)
                fail += 1

        out_path = args.output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)
        print(f"  {stem}: {ok} enriched, {fail} kept original -> {out_path}")

    print(
        "\nNext: run_daily_summaries_by_symbol --news-dir",
        str(args.output_dir),
        "--output-dir data/te_commodities/mc_daily_summaries",
    )
    print("Then: merge_all_te_text --summaries-dir data/te_commodities/mc_daily_summaries --data-dir data/te_commodities")


if __name__ == "__main__":
    main()
