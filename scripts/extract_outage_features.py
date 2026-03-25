"""Extract numeric outage features from the text column of hydro reservoir CSVs.

Parses PLANNED/UNPLANNED sections and adds four columns:
  planned_count, planned_mw, unplanned_count, unplanned_mw
"""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

_COUNT_RE = re.compile(
    r"(\d+|" + "|".join(WORD_TO_NUM) + r")\s+"
    r"(?:planned|scheduled|forced|unplanned)?\s*outages?",
    re.IGNORECASE,
)

# "totalling 185 MW", "removing 68 MW", "of 110 MW", "remove 200 MW",
# "removed 77 MW", "(100 MW)", ", 75 MW unavailable", ", 320 MW unavailable"
_MW_RE = re.compile(
    r"(?:totalling|removing|remove[ds]?|of)\s+([\d\s,]+)\s*MW"
    r"|,\s*([\d\s,]+)\s*MW\s+(?:unavailable|offline)",
    re.IGNORECASE,
)

# Standalone pattern for "N outage(s), X MW unavailable" or "N outage (X MW)"
_MW_PAREN_RE = re.compile(r"\((\d[\d\s,]*)\s*MW\)")


def _parse_mw(raw: str) -> float:
    """Parse an MW string that may contain comma or space thousands separators."""
    cleaned = re.sub(r"[\s\u00a0\u202f,]+", "", raw).strip()
    return float(cleaned)


def _parse_count(raw: str) -> int:
    raw_lower = raw.strip().lower()
    if raw_lower in WORD_TO_NUM:
        return WORD_TO_NUM[raw_lower]
    return int(raw_lower)


def _extract_section(section_text: str) -> tuple[int, float]:
    """Return (count, total_mw) for a single PLANNED or UNPLANNED section."""
    stripped = section_text.strip()
    if not stripped or stripped.rstrip(".").strip().lower() == "none":
        return 0, 0.0

    count_match = _COUNT_RE.search(stripped)
    count = _parse_count(count_match.group(1)) if count_match else 1

    mw_match = _MW_RE.search(stripped)
    if mw_match:
        mw_str = mw_match.group(1) or mw_match.group(2)
        mw = _parse_mw(mw_str)
    else:
        paren_match = _MW_PAREN_RE.search(stripped)
        if paren_match:
            mw = _parse_mw(paren_match.group(1))
        else:
            mw = 0.0

    return count, mw


def parse_outage_text(text: str | float) -> dict[str, float]:
    if not isinstance(text, str) or not text.strip():
        return {
            "planned_count": np.nan,
            "planned_mw": np.nan,
            "unplanned_count": np.nan,
            "unplanned_mw": np.nan,
        }

    planned_count, planned_mw = 0, 0.0
    unplanned_count, unplanned_mw = 0, 0.0

    planned_match = re.search(r"PLANNED:\s*(.*?)(?=UNPLANNED:|$)", text, re.DOTALL)
    unplanned_match = re.search(r"UNPLANNED:\s*(.*?)$", text, re.DOTALL)

    if planned_match:
        planned_count, planned_mw = _extract_section(planned_match.group(1))
    if unplanned_match:
        unplanned_count, unplanned_mw = _extract_section(unplanned_match.group(1))

    return {
        "planned_count": float(planned_count),
        "planned_mw": planned_mw,
        "unplanned_count": float(unplanned_count),
        "unplanned_mw": unplanned_mw,
    }


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    files = sorted(glob(str(data_dir / "*_daily_hydro_reservoir.csv")))
    if not files:
        print("No *_daily_hydro_reservoir.csv files found in", data_dir)
        return

    for fpath in files:
        p = Path(fpath)
        fname = p.name
        out_path = p.with_name(p.stem + "_features.csv")

        df = pd.read_csv(fpath)
        features = df["text"].apply(parse_outage_text).apply(pd.Series)
        for col in features.columns:
            df[col] = features[col]

        df.to_csv(out_path, index=False)

        n_total = len(df)
        n_with_text = df["planned_count"].notna().sum()
        n_nan = df["planned_count"].isna().sum()
        print(f"\n{'='*60}")
        print(f"  {fname} -> {out_path.name}")
        print(f"  rows: {n_total}  |  parsed: {n_with_text}  |  empty text (NaN): {n_nan}")
        print(f"  planned_mw  — min: {df['planned_mw'].min():.0f}  max: {df['planned_mw'].max():.0f}  mean: {df['planned_mw'].mean():.1f}")
        print(f"  unplanned_mw — min: {df['unplanned_mw'].min():.0f}  max: {df['unplanned_mw'].max():.0f}  mean: {df['unplanned_mw'].mean():.1f}")
        sample = df[df["planned_count"].notna()].head(3)[["t", "planned_count", "planned_mw", "unplanned_count", "unplanned_mw"]]
        print(sample.to_string(index=False))
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
