#!/usr/bin/env python3
"""
Resolve Trading Economics symbol meanings (human-readable names) for series in
data/te_countries/with_text/.

Reads CSV filenames from --with-text-dir (pattern: {Country}_{Symbol}.csv),
fetches names from TE API where possible, and writes a JSON/CSV mapping.

Sources:
  - /markets/country/{country} returns Symbol (e.g. SPX:IND), Name for all
    instruments in that country. We match by ticker (part before ":" in Symbol).
  - Optional: /forecast/country/{countries} for indicator Category (name) by
    HistoricalDataSymbol.

Requires: TRADING_ECONOMICS_API_KEY or TE_API_KEY for API lookups.
Output: symbol_meanings.json and symbol_meanings.csv in --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import requests


# Fallback names when API is not used or symbol not found (common TE tickers in this repo)
FALLBACK_NAMES: dict[str, str] = {
    # United States
    "SPX": "S&P 500",
    "DXY": "US Dollar Index",
    "USGG10YR": "United States 10-Year Government Bond Yield",
    "IJCUSA": "Initial Jobless Claims",
    "UNITEDSTACRUOILSTOCH": "United States Crude Oil Stocks",
    "UNITEDSTAMORAPP": "United States Mortgage Applications",
    # UK
    "GUKG10": "United Kingdom 10-Year Government Bond Yield",
    "GBPUSD": "British Pound vs US Dollar",
    "UKX": "FTSE 100",
    # Germany
    "DAX": "DAX",
    "GDBR10": "Germany 10-Year Government Bond Yield",
    # France
    "CAC": "CAC 40",
    "GFRN10": "France 10-Year Government Bond Yield",
    # Italy
    "FTSEMIB": "FTSE MIB",
    "GBTPGR10": "Italy 10-Year Government Bond Yield",
    # Spain
    "IBEX": "IBEX 35",
    # Canada
    "GCAN10YR": "Canada 10-Year Government Bond Yield",
    "USDCAD": "Canadian Dollar vs US Dollar",
    "SPTSX": "S&P/TSX Composite",
    # Mexico
    "USDMXN": "Mexican Peso vs US Dollar",
    # Commodities
    "CL1": "Crude Oil WTI",
    "CO1": "Brent Crude Oil",
    "XAUUSD": "Gold",
    "XAGUSD": "Silver",
    "XPDUSD": "Palladium",
    "NG1": "Natural Gas (Henry Hub)",
    "NGEU": "Natural Gas Europe",
    "NGUK": "Natural Gas UK",
    "KC1": "Coffee",
    "CC1": "Cocoa",
    "SB1": "Sugar",
    "W_1": "Wheat",
    "W 1": "Wheat",
    "W1": "Wheat",
    "S_1": "Soybeans",
    "S 1": "Soybeans",
    "S1": "Soybeans",
    "XB1": "Cotton",
    "HO1": "Heating Oil",
    "HG1": "Copper",
    "BALTIC": "Baltic Dry Index",
    "PALMOIL": "Palm Oil",
    "Steel": "Steel",
}


def get_api_key() -> str | None:
    return os.environ.get("TRADING_ECONOMICS_API_KEY") or os.environ.get("TE_API_KEY")


# Country name as used in filenames (with underscores). Used to split stem into country + symbol.
_COUNTRY_STEMS = [
    "United_States",
    "United_Kingdom",
    "Commodity",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Canada",
    "Mexico",
]


def stems_from_with_text_dir(with_text_dir: Path) -> list[tuple[str, str]]:
    """Return list of (country, symbol) from filenames Country_Symbol.csv."""
    pairs: list[tuple[str, str]] = []
    for p in with_text_dir.glob("*.csv"):
        stem = p.stem
        # Symbol may contain underscores (e.g. W_1, S_1), so split on known country prefix
        country_stem = None
        for c in _COUNTRY_STEMS:
            if stem == c or stem.startswith(c + "_"):
                country_stem = c
                break
        if not country_stem:
            # Fallback: last underscore (wrong for W_1, S_1)
            idx = stem.rfind("_")
            if idx <= 0:
                continue
            country = stem[:idx].replace("_", " ")
            symbol = stem[idx + 1 :]
        else:
            country = country_stem.replace("_", " ")
            symbol = stem[len(country_stem) + 1 :]  # rest is symbol (may have underscores)
        pairs.append((country, symbol))
    return pairs


def fetch_markets_by_country(api_key: str, country: str) -> list[dict]:
    """GET /markets/country/{country}. Returns list of {Symbol, Name, Country, Type, ...}."""
    slug = country.lower().replace(" ", "%20")
    url = f"https://api.tradingeconomics.com/markets/country/{slug}"
    params = {"c": api_key, "f": "json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def ticker_from_te_symbol(te_symbol: str) -> str:
    """TE Symbol is e.g. SPX:IND, CL1:COM, A:US. Return ticker (part before colon) or full."""
    s = (te_symbol or "").strip()
    if ":" in s:
        return s.split(":")[0].strip()
    return s


def fetch_forecast_indicators(api_key: str, countries: list[str]) -> list[dict]:
    """GET /forecast/country/{countries}. Returns indicators with HistoricalDataSymbol, Category."""
    slug_list = [c.lower().replace(" ", "%20") for c in countries]
    country_path = ",".join(slug_list)
    url = f"https://api.tradingeconomics.com/forecast/country/{country_path}"
    params = {"c": api_key, "f": "json"}
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def build_meanings(
    pairs: list[tuple[str, str]],
    api_key: str | None,
    use_forecast: bool,
    sleep: float,
) -> dict[str, str]:
    """
    Build stem -> name. stem = "Country_Symbol" (underscores as in filename).
    """
    stem_to_name: dict[str, str] = {}
    # Apply fallbacks first (so we always have something for known symbols)
    for country, symbol in pairs:
        stem = f"{country.replace(' ', '_')}_{symbol.replace(' ', '_')}"
        name = FALLBACK_NAMES.get(symbol) or FALLBACK_NAMES.get(symbol.replace(" ", "_"))
        if name:
            stem_to_name[stem] = name

    if not api_key:
        return stem_to_name

    # Unique countries (including "Commodity")
    countries = sorted({c for c, _ in pairs})
    # Markets by country -> (country, ticker) -> name
    for country in countries:
        try:
            if sleep > 0:
                time.sleep(sleep)
            rows = fetch_markets_by_country(api_key, country)
        except Exception as e:
            print(f"  Warning: markets/country/{country}: {e}")
            continue
        for row in rows:
            te_symbol = (row.get("Symbol") or row.get("symbol") or "").strip()
            name = (row.get("Name") or row.get("name") or "").strip()
            if not name:
                continue
            ticker = ticker_from_te_symbol(te_symbol)
            if not ticker:
                continue
            # Match our (country, symbol) pairs
            for c, sym in pairs:
                if c != country:
                    continue
                # Normalize: our symbol can be "W 1" or "W_1"
                sym_norm = sym.replace(" ", "_")
                ticker_norm = ticker.replace(" ", "_")
                if sym == ticker or sym_norm == ticker_norm:
                    stem = f"{c.replace(' ', '_')}_{sym.replace(' ', '_')}"
                    stem_to_name[stem] = name  # API name overrides fallback
                    break

    # Forecast API: (country, HistoricalDataSymbol) -> Category
    if use_forecast and api_key:
        try:
            # Only country-level (exclude Commodity for forecast)
            forecast_countries = [c for c in countries if c != "Commodity"]
            if forecast_countries and sleep > 0:
                time.sleep(sleep)
            indicators = fetch_forecast_indicators(api_key, forecast_countries)
        except Exception as e:
            print(f"  Warning: forecast/country: {e}")
            indicators = []
        for ind in indicators:
            country = (ind.get("Country") or ind.get("country") or "").strip()
            sym = (ind.get("HistoricalDataSymbol") or "").strip()
            cat = (ind.get("Category") or ind.get("category") or ind.get("Title") or "").strip()
            if not country or not sym or not cat:
                continue
            for c, symbol in pairs:
                if c != country:
                    continue
                if symbol == sym or symbol.replace(" ", "_") == sym.replace(" ", "_"):
                    stem = f"{c.replace(' ', '_')}_{symbol.replace(' ', '_')}"
                    stem_to_name[stem] = cat
                    break

    return stem_to_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve TE symbol meanings for with_text series",
    )
    parser.add_argument(
        "--with-text-dir",
        type=Path,
        default=Path("data/te_countries/with_text"),
        help="Directory containing Country_Symbol.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries"),
        help="Write symbol_meanings.json and symbol_meanings.csv here",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Use only fallback names (no API calls)",
    )
    parser.add_argument(
        "--no-forecast",
        action="store_true",
        help="Do not call forecast API for indicator names",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds between API requests",
    )
    args = parser.parse_args()

    if not args.with_text_dir.exists():
        raise SystemExit(f"Directory not found: {args.with_text_dir}")

    pairs = stems_from_with_text_dir(args.with_text_dir)
    if not pairs:
        print("No CSV files found.")
        return

    api_key = None if args.no_api else get_api_key()
    if not api_key and not args.no_api:
        print("No TE_API_KEY set; using fallback names only. Set TRADING_ECONOMICS_API_KEY or TE_API_KEY for API lookups.")

    stem_to_name = build_meanings(
        pairs,
        api_key=api_key,
        use_forecast=not args.no_forecast,
        sleep=args.sleep,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "symbol_meanings.json"
    out_csv = args.output_dir / "symbol_meanings.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stem_to_name, f, indent=2)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "name"])
        for stem in sorted(stem_to_name):
            w.writerow([stem, stem_to_name[stem]])

    print(f"Wrote {out_json} and {out_csv} ({len(stem_to_name)} entries).")
    missing = [f"{c}_{s}".replace(" ", "_") for c, s in pairs if f"{c.replace(' ', '_')}_{s.replace(' ', '_')}" not in stem_to_name]
    if missing:
        print(f"Missing names for: {missing}")


if __name__ == "__main__":
    main()
