#!/usr/bin/env python3
"""
Slice with_text CSVs by the period spec and write to with_text_most_annotated.

Reads data/te_countries/period_spec.json (country, symbol, periods). For each entry,
loads the corresponding CSV from with_text/, filters rows by the specified date range(s),
and writes to with_text_most_annotated/. When multiple periods are given (e.g. ">=2017 <=2018; >=2021"),
writes multiple CSVs with suffix _2017-2018.csv and _2021-onwards.csv.

Period format:
  - "weekly full" or "fine" → use entire file
  - ">=YYYY" or ">=YYYY good" → from YYYY-01-01 onwards
  - ">=YYYY-MM-DD" → from that date onwards
  - ">=YYYY <=YYYY" → from YYYY1-01-01 through YYYY2-12-31
  - ">=2017 <=2018; >=2021" → two segments: [2017-01-01, 2018-12-31] and [2021-01-01, end]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def sanitize_stem(country: str, symbol: str) -> str:
    c = re.sub(r"[^\w\s]", "", country).strip().replace(" ", "_")
    s = re.sub(r"[^\w]", "_", symbol.strip())
    return f"{c}_{s}"


def find_source_csv(source_dir: Path, country: str, symbol: str) -> Path | None:
    stem = sanitize_stem(country, symbol)
    p = source_dir / f"{stem}.csv"
    if p.exists():
        return p
    # Try symbol with underscore before trailing digits (e.g. W1 -> W_1, S1 -> S_1)
    alt = re.sub(r"^(\D)(\d+)$", r"\1_\2", symbol.strip())
    if alt != symbol:
        stem_alt = sanitize_stem(country, alt)
        p = source_dir / f"{stem_alt}.csv"
        if p.exists():
            return p
    return None


def parse_periods(s: str) -> list[tuple[str | None, str | None]]:
    """
    Parse period string into list of (start_date, end_date).
    start_date/end_date are YYYY-MM-DD or None (no bound).
    """
    s = (s or "").strip().lower()
    # Strip trailing " good" or similar
    s = re.sub(r"\s+good\s*$", "", s).strip()
    if not s or s in ("weekly full", "fine", "full"):
        return [(None, None)]

    out = []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for part in parts:
        start_date = None
        end_date = None
        m = re.search(r">=\s*(\d{4})(?:-(\d{2})-(\d{2}))?", part)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            start_date = f"{y}-{mo or '01'}-{d or '01'}"
        m2 = re.search(r"<=\s*(\d{4})(?:-(\d{2})-(\d{2}))?", part)
        if m2:
            y, mo, d = m2.group(1), m2.group(2), m2.group(3)
            end_date = f"{y}-{mo or '12'}-{d or '31'}" if (mo and d) else f"{y}-12-31"
        out.append((start_date, end_date))
    return out


def period_to_suffix(start: str | None, end: str | None, total: int) -> str:
    """Generate filename suffix for a period (e.g. _2017-2018, _2021-onwards). Empty if single period."""
    if start is None and end is None:
        return ""
    if total > 1:
        if start and end:
            return f"_{start[:4]}-{end[:4]}"
        if start:
            return f"_{start[:4]}-onwards"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice with_text CSVs by period_spec into with_text_most_annotated")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("data/te_countries/period_spec.json"),
        help="JSON spec: list of {country, symbol, periods}",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/te_countries/with_text"),
        help="Source directory of CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries/with_text_most_annotated"),
        help="Output directory for sliced CSVs",
    )
    args = parser.parse_args()

    if not args.spec.exists():
        raise SystemExit(f"Spec not found: {args.spec}")
    if not args.source_dir.is_dir():
        raise SystemExit(f"Source dir not found: {args.source_dir}")

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, list):
        spec = [spec]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for entry in spec:
        country = str(entry.get("country", "")).strip()
        symbol = str(entry.get("symbol", "")).strip()
        periods_str = str(entry.get("periods", "")).strip()
        if not country or not symbol:
            continue

        csv_path = find_source_csv(args.source_dir, country, symbol)
        if not csv_path:
            print(f"  Skip {country} / {symbol}: no source CSV found")
            continue

        df = pd.read_csv(csv_path)
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
        if df.empty:
            print(f"  Skip {country} / {symbol}: empty or invalid dates")
            continue

        base_stem = csv_path.stem
        parsed = parse_periods(periods_str)
        if len(parsed) == 1 and parsed[0][0] is None and parsed[0][1] is None:
            out_path = args.output_dir / f"{base_stem}.csv"
            df_out = df.sort_values("t")
            df_out["t"] = df_out["t"].dt.strftime("%Y-%m-%d")
            df_out.to_csv(out_path, index=False, quoting=1)
            print(f"  {base_stem}: full -> {out_path.name}")
            continue

        for idx, (start, end) in enumerate(parsed):
            mask = pd.Series(True, index=df.index)
            if start:
                mask &= df["t"] >= pd.Timestamp(start)
            if end:
                mask &= df["t"] <= pd.Timestamp(end)
            df_out = df.loc[mask].sort_values("t")
            if df_out.empty:
                print(f"  {base_stem}: period {start} to {end} -> no rows, skip")
                continue
            suffix = period_to_suffix(start, end, len(parsed))
            out_name = f"{base_stem}{suffix}.csv"
            out_path = args.output_dir / out_name
            df_out = df_out.copy()
            df_out["t"] = df_out["t"].dt.strftime("%Y-%m-%d")
            df_out.to_csv(out_path, index=False, quoting=1)
            print(f"  {base_stem}: {start or '..'} to {end or '..'} -> {out_path.name} ({len(df_out)} rows)")

    print(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
