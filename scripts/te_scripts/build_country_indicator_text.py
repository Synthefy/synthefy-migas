#!/usr/bin/env python3
"""
Build date -> text from TE raw news per (country, symbol) and merge into numeric CSVs.

Reads raw news from te_news_raw_old, aggregates by period (month/year/quarter) to match
series frequency, then merges into raw CSVs (from fetch_country_indicator_historical.py)
and writes final TTFM CSVs (t, y_t, text) to data/te_countries/with_text/.

Uses same trim-leading-empty-text logic as scripts/merge_text_numerical.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_country_config(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load te_countries.json; return (display_names list, alias_lower -> display_name)."""
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


def load_raw_news_by_country_symbol(
    raw_dir: Path,
    alias_to_display: dict[str, str],
    allowed_display_names: set[str],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """(country_display, symbol) -> list of (date_yyyy_mm_dd, text)."""
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


def period_end_for_date(date_str: str, frequency: str) -> str:
    """Map a news date to the period end string used in the series (YYYY-MM-DD)."""
    try:
        dt = pd.Timestamp(date_str)
    except Exception:
        return ""
    freq = (frequency or "").strip().lower()
    if freq == "monthly":
        return (dt + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    if freq == "yearly":
        return f"{dt.year}-12-31"
    if freq == "quarterly":
        return (dt + pd.offsets.QuarterEnd(0)).strftime("%Y-%m-%d")
    if freq == "weekly":
        # Week ending Saturday (same as common US reporting)
        days_to_sat = (5 - dt.dayofweek + 7) % 7
        if days_to_sat == 0 and dt.dayofweek != 5:
            days_to_sat = 7
        return (dt + pd.Timedelta(days=days_to_sat)).strftime("%Y-%m-%d")
    return (dt + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")


def dates_in_period(period_end_str: str, frequency: str) -> list[str]:
    """Return list of YYYY-MM-DD dates that fall in the period ending on period_end_str."""
    try:
        end_dt = pd.Timestamp(period_end_str)
    except Exception:
        return []
    freq = (frequency or "").strip().lower()
    if freq == "monthly":
        start_dt = end_dt.replace(day=1)
        dr = pd.date_range(start=start_dt, end=end_dt, freq="D")
        return [d.strftime("%Y-%m-%d") for d in dr]
    if freq == "yearly":
        start_dt = end_dt.replace(month=1, day=1)
        dr = pd.date_range(start=start_dt, end=end_dt, freq="D")
        return [d.strftime("%Y-%m-%d") for d in dr]
    if freq == "quarterly":
        q = (end_dt.month - 1) // 3 + 1
        start_dt = pd.Timestamp(year=end_dt.year, month=(q - 1) * 3 + 1, day=1)
        dr = pd.date_range(start=start_dt, end=end_dt, freq="D")
        return [d.strftime("%Y-%m-%d") for d in dr]
    if freq == "weekly":
        # Week ending Saturday: start = end - 6 days
        start_dt = end_dt - pd.Timedelta(days=6)
        dr = pd.date_range(start=start_dt, end=end_dt, freq="D")
        return [d.strftime("%Y-%m-%d") for d in dr]
    if freq == "daily":
        return [period_end_str]
    # default monthly
    start_dt = end_dt.replace(day=1)
    dr = pd.date_range(start=start_dt, end=end_dt, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dr]


def build_date_to_text_from_daily_summaries(
    summaries_path: Path,
    frequency: str,
    numeric_dates: pd.Series,
) -> dict[str, str]:
    """Load daily summaries JSON; aggregate by period to match numeric CSV dates. Returns period_end -> text."""
    if not summaries_path.exists():
        return {}
    with open(summaries_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}
    date_to_summary = {}
    for item in data:
        d = item.get("date") or item.get("Date")
        s = (item.get("daily_summary") or "").strip()
        if d and s:
            date_to_summary[d] = s
    period_to_texts: dict[str, list[str]] = defaultdict(list)
    for t_val in numeric_dates.dropna().unique():
        if hasattr(t_val, "strftime"):
            period_end = t_val.strftime("%Y-%m-%d")
        else:
            period_end = str(pd.Timestamp(t_val).date())
        for d in dates_in_period(period_end, frequency):
            if d in date_to_summary:
                period_to_texts[period_end].append(date_to_summary[d])
    return {
        k: "\n\n---\n\n".join(v) if v else "NA"
        for k, v in period_to_texts.items()
    }


def build_date_to_text(
    entries: list[tuple[str, str]],
    frequency: str,
) -> dict[str, str]:
    """Aggregate (date, text) by period end; return period_end_yyyy_mm_dd -> concatenated text."""
    period_to_texts: dict[str, list[str]] = defaultdict(list)
    for date_str, text in entries:
        key = period_end_for_date(date_str, frequency)
        if key:
            period_to_texts[key].append(text)
    return {
        k: "\n\n---\n\n".join(v) if v else "NA"
        for k, v in period_to_texts.items()
    }


def merge_and_trim(
    df: pd.DataFrame,
    date_to_text: dict[str, str],
) -> pd.DataFrame:
    """Add text column by date lookup; trim leading rows without text."""
    df = df.copy()
    if "frequency" in df.columns:
        df = df.drop(columns=["frequency"])
    text_values = []
    for _, row in df.iterrows():
        t_val = row["t"]
        if hasattr(t_val, "strftime"):
            date_str = t_val.strftime("%Y-%m-%d")
        else:
            date_str = str(pd.Timestamp(t_val).date())
        text_values.append(date_to_text.get(date_str, "NA"))
    df["text"] = text_values

    first_text_idx = None
    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        if text and text != "NA" and text != "":
            first_text_idx = idx
            break
    if first_text_idx is not None:
        df = df.loc[first_text_idx:].copy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge raw news text into country-indicator CSVs; output TTFM (t, y_t, text) CSVs",
    )
    parser.add_argument(
        "--raw-news-dir",
        type=Path,
        default=Path("data/te_commodities/te_news_raw_old"),
        help="Directory of monthly YYYY-MM.json raw news",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/te_countries/raw"),
        help="Directory of numeric CSVs and manifest.json from fetch_country_indicator_historical",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries/with_text"),
        help="Directory for final CSVs (t, y_t, text)",
    )
    parser.add_argument(
        "--countries-json",
        type=Path,
        default=Path("data/te_countries/te_countries.json"),
        help="Country config for normalizing names",
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=None,
        help="If set, use daily summaries JSONs from this dir (from create_daily_summaries) instead of raw news.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Process only this stem (e.g. United_States_USURTOT); default: all in manifest",
    )
    args = parser.parse_args()

    use_summaries = args.summaries_dir is not None and args.summaries_dir.is_dir()
    if not use_summaries and not args.raw_news_dir.is_dir():
        raise SystemExit(f"Raw news dir not found: {args.raw_news_dir}. Or set --summaries-dir for vLLM summaries.")
    if not args.raw_dir.is_dir():
        raise SystemExit(f"Raw dir not found: {args.raw_dir}")

    display_names, alias_to_display = load_country_config(args.countries_json)
    allowed = set(display_names)

    if use_summaries:
        print(f"Using daily summaries from {args.summaries_dir}")
    else:
        print("Loading raw news ...")
        news_by_pair = load_raw_news_by_country_symbol(
            args.raw_news_dir, alias_to_display, allowed
        )
        print(f"  Loaded {len(news_by_pair)} (country, symbol) pairs.")

    manifest_path = args.raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run fetch_country_indicator_historical first.")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        stem = entry.get("stem", "")
        country = entry.get("country", "")
        symbol = entry.get("symbol", "")
        frequency = entry.get("frequency", "Monthly")
        if args.stem and stem != args.stem:
            continue
        csv_path = args.raw_dir / f"{stem}.csv"
        if not csv_path.exists():
            print(f"Skip {stem}: CSV not found")
            continue
        df = pd.read_csv(csv_path)
        df["t"] = pd.to_datetime(df["t"])
        if use_summaries:
            summaries_path = args.summaries_dir / f"{stem}.json"
            date_to_text = build_date_to_text_from_daily_summaries(
                summaries_path, frequency, df["t"]
            )
        else:
            key = (country, symbol)
            entries_list = news_by_pair.get(key, [])
            date_to_text = build_date_to_text(entries_list, frequency)

        out_df = merge_and_trim(df, date_to_text)
        out_df["t"] = out_df["t"].dt.strftime("%Y-%m-%d")
        out_path = args.output_dir / f"{stem}.csv"
        out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        n_text = (out_df["text"] != "NA").sum()
        print(f"  {stem}: {len(out_df)} rows, {n_text} with text -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
