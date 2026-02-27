#!/usr/bin/env python3
"""
Fetch the list of commodities from Trading Economics API.

Calls GET /markets/commodities, saves the list (JSON and optionally CSV),
and prints symbols + names for hand-picking US-relevant commodities.
API key: TRADING_ECONOMICS_API_KEY or TE_API_KEY.
"""

import argparse
import json
import os
from pathlib import Path

import requests


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("TRADING_ECONOMICS_API_KEY") or os.environ.get("TE_API_KEY")
    if not key:
        raise SystemExit(
            "Error: Set TRADING_ECONOMICS_API_KEY or TE_API_KEY in the environment."
        )
    return key


def fetch_commodities(api_key: str) -> list:
    """Fetch commodities snapshot from Trading Economics."""
    url = "https://api.tradingeconomics.com/markets/commodities"
    params = {"c": api_key, "f": "json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch commodity list from Trading Economics and save for selection"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/te_commodities",
        help="Directory to write commodities list JSON (default: data/te_commodities)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a CSV with Symbol, Name, Country, Group, etc.",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    print("Fetching commodities from Trading Economics...")
    data = fetch_commodities(api_key)

    if not data:
        print("No commodities returned.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "commodities_list.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} commodities to {json_path}")

    if args.csv:
        import csv as csv_module

        csv_path = out_dir / "commodities_list.csv"
        if data:
            keys = list(data[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv_module.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(data)
            print(f"Saved CSV to {csv_path}")

    print("\nSymbols and names (for hand-picking US-relevant commodities):")
    print("-" * 60)
    for item in data:
        symbol = item.get("Symbol", "")
        name = item.get("Name", "")
        country = item.get("Country", "")
        group = item.get("Group", "")
        print(f"  {symbol:<20} {name:<35} country={country} group={group}")
    print("-" * 60)
    print(f"Total: {len(data)} commodities. Edit te_commodity_symbols.json to select 20-30.")


if __name__ == "__main__":
    main()
