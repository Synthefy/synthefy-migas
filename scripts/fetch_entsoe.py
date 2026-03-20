"""
ENTSO-E Data Fetcher
====================
Pulls two datasets from the ENTSO-E Transparency Platform REST API and joins them:

1. TARGET TIME SERIES — Day-ahead electricity prices (hourly) for a chosen bidding zone
2. TEXT COVARIATES   — REMIT Urgent Market Messages (outage notices with free-text fields)

Prerequisites:
  - Register at https://transparency.entsoe.eu/ and request an API token
    (Settings > API token after login)
  - pip install requests pandas lxml

Usage:
  python fetch_entsoe.py

Output:
  - entsoe_prices_{zone}.csv          — hourly day-ahead prices
  - entsoe_remit_{zone}.csv           — REMIT outage messages with text fields
  - entsoe_joined_{zone}.csv          — merged dataset (hourly price + aligned text events)

Notes:
  - The ENTSO-E API returns XML; we parse with lxml
  - REMIT messages are event-level (start/end); we explode them to hourly rows for the join
  - Free tier has rate limits (~400 requests/min). This script stays well within that.
  - Adjust ZONE, START, END below for your needs.
"""

import io
import zipfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import os
import sys

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ══════════════════════════════════════════════════════════════════════════════

API_TOKEN = os.environ.get("ENTSOE_API_TOKEN", "YOUR_TOKEN_HERE")

# Bidding zone EIC codes — common ones:
#   Germany-Luxembourg: 10Y1001A1001A82H
#   France:             10YFR-RTE------C
#   Netherlands:        10YNL----------L
#   Spain:              10YES-REE------0
#   Italy North:        10Y1001A1001A73I
ZONE = "10Y1001A1001A82H"  # Germany-Luxembourg
ZONE_LABEL = "DE_LU"

# Date range (UTC, format: YYYYMMDD0000)
# ENTSO-E API allows max ~1 year per request for prices
START = "202401010000"
END   = "202412310000"

BASE_URL = "https://web-api.tp.entsoe.eu/api"

OUTPUT_DIR = "."


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — generic API call with retry
# ══════════════════════════════════════════════════════════════════════════════

TOO_MANY = "TOO_MANY"  # sentinel: window too large, caller should halve it


def call_entsoe(params: dict, description: str):
    """
    Call the ENTSO-E API and return XML text / list[str] for ZIP responses.
    Returns TOO_MANY sentinel if the API says the window has too many instances.
    Returns None on other failures.
    """
    params = {**params, "securityToken": API_TOKEN}

    for attempt in range(3):
        try:
            print(f"  → Fetching {description} (attempt {attempt+1})...")
            resp = requests.get(BASE_URL, params=params, timeout=120)

            if resp.status_code == 200:
                if resp.content[:2] == b"PK":
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        names = [n for n in zf.namelist() if n.endswith(".xml")]
                        if not names:
                            return None
                        return [zf.read(n).decode("utf-8") for n in names]
                return resp.text
            elif resp.status_code == 401:
                print("  ✗ 401 Unauthorized — check your API token")
                sys.exit(1)
            elif resp.status_code == 429:
                print("  ⏳ Rate limited, waiting 60s...")
                time.sleep(60)
            else:
                try:
                    root = ET.fromstring(resp.text)
                    reason = next(
                        (el.text for el in root.iter() if el.tag.endswith("}text") and el.text),
                        resp.text[:200],
                    )
                    if "exceeds the allowed maximum" in (reason or ""):
                        print(f"  ↓ Too many instances — halving window")
                        return TOO_MANY  # don't retry, just shrink
                    print(f"  ✗ HTTP {resp.status_code}: {reason}")
                except Exception:
                    print(f"  ✗ HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request error: {e}")
            time.sleep(5)

    print(f"  ✗ Failed to fetch {description} after 3 attempts")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. DAY-AHEAD PRICES (document type A44)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_day_ahead_prices(zone: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly day-ahead prices for a bidding zone.
    Document type A44 = Price Document
    """
    print("\n[1/3] Fetching day-ahead prices...")

    # Split into monthly chunks to avoid API limits
    start_dt = datetime.strptime(start, "%Y%m%d%H%M")
    end_dt = datetime.strptime(end, "%Y%m%d%H%M")

    all_rows = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=31), end_dt)
        chunk_end = chunk_end.replace(day=1)  # align to month boundary
        if chunk_end <= chunk_start:
            chunk_end = end_dt

        params = {
            "documentType": "A44",
            "in_Domain": zone,
            "out_Domain": zone,
            "periodStart": chunk_start.strftime("%Y%m%d%H%M"),
            "periodEnd": chunk_end.strftime("%Y%m%d%H%M"),
        }

        xml_text = call_entsoe(params, f"prices {chunk_start.strftime('%Y-%m')}")

        if xml_text:
            rows = parse_price_xml(xml_text)
            all_rows.extend(rows)
            print(f"    ✓ {len(rows)} hourly price points")

        chunk_start = chunk_end
        time.sleep(1)  # be polite

    if not all_rows:
        print("  ✗ No price data retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp_utc", "price_eur_mwh"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)

    print(f"  ✓ Total: {len(df)} hourly prices, {df.timestamp_utc.min()} → {df.timestamp_utc.max()}")
    return df


def parse_price_xml(xml_text: str) -> list:
    """Parse ENTSO-E price XML into (timestamp, price) tuples."""
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
    rows = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return rows

    for ts in root.findall(".//ns:TimeSeries", ns):
        for period in ts.findall("ns:Period", ns):
            start_el = period.find("ns:timeInterval/ns:start", ns)
            res_el = period.find("ns:resolution", ns)

            if start_el is None:
                continue

            period_start = datetime.strptime(start_el.text, "%Y-%m-%dT%H:%MZ")

            for point in period.findall("ns:Point", ns):
                pos = int(point.find("ns:position", ns).text)
                price_el = point.find("ns:price.amount", ns)
                if price_el is not None:
                    ts_utc = period_start + timedelta(hours=pos - 1)
                    rows.append((ts_utc, float(price_el.text)))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 2. REMIT URGENT MARKET MESSAGES (document type A78)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_remit_messages(zone: str, start: str, end: str, incremental_path: str = None) -> pd.DataFrame:
    """
    Fetch REMIT Urgent Market Messages (UMMs) for a bidding zone.
    Document type A78 = Unavailability of generation/production units.

    These contain FREE-TEXT fields:
      - reason text (why the outage happened)
      - remarks
      - asset/unit name
      - fuel type
      - available/installed capacity
    """
    print("\n[2/3] Fetching REMIT messages...")

    # Adaptive chunking: start at 2h, halve on TOO_MANY, double back after success
    start_dt = datetime.strptime(start, "%Y%m%d%H%M")
    end_dt = datetime.strptime(end, "%Y%m%d%H%M")

    all_messages = []
    _header_written = False
    chunk_start = start_dt
    chunk_hours = 2.0

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(hours=chunk_hours), end_dt)

        for doc_type in ["A78", "A77"]:
            window_start = chunk_start
            window_end   = chunk_end
            window_hours = chunk_hours

            while True:
                if doc_type == "A77":
                    params = {
                        "documentType": doc_type,
                        "biddingZone_Domain": zone,
                        "periodStart": window_start.strftime("%Y%m%d%H%M"),
                        "periodEnd":   window_end.strftime("%Y%m%d%H%M"),
                    }
                else:
                    params = {
                        "documentType": doc_type,
                        "in_Domain": zone,
                        "out_Domain": zone,
                        "periodStart": window_start.strftime("%Y%m%d%H%M"),
                        "periodEnd":   window_end.strftime("%Y%m%d%H%M"),
                    }
                result = call_entsoe(params, f"REMIT {doc_type} {window_start.strftime('%Y-%m-%d %H:%M')} ({window_hours:.1f}h)")

                if result is TOO_MANY:
                    window_hours /= 2
                    if window_hours < 0.25:  # give up below 15-min window
                        print(f"  ✗ Window too small, skipping {window_start}")
                        break
                    window_end = window_start + timedelta(hours=window_hours)
                    continue

                if result:
                    xml_list = result if isinstance(result, list) else [result]
                    chunk_msgs = [m for x in xml_list for m in parse_remit_xml(x, doc_type)]
                    all_messages.extend(chunk_msgs)
                    if incremental_path and chunk_msgs:
                        chunk_df = pd.DataFrame(chunk_msgs)
                        chunk_df.to_csv(incremental_path, mode="a", index=False,
                                        header=not _header_written)
                        _header_written = True
                    print(f"    ✓ {len(chunk_msgs)} messages ({doc_type})")
                break

            time.sleep(1)

        chunk_start = chunk_end

    if not all_messages:
        print("  ✗ No REMIT messages retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_messages)
    df["event_start"] = pd.to_datetime(df["event_start"])
    df["event_end"] = pd.to_datetime(df["event_end"])
    df = df.sort_values("event_start").reset_index(drop=True)

    print(f"  ✓ Total: {len(df)} REMIT messages")
    print(f"    Columns: {list(df.columns)}")
    return df


def parse_remit_xml(xml_text: str, doc_type: str) -> list:
    """
    Parse a single REMIT UMM XML file (one TimeSeries = one outage event).

    Field mapping (from actual ENTSO-E response structure):
      - businessType            → event_type  (A53=planned, A54=unplanned)
      - production_RegisteredResource.name           → asset_name
      - production_RegisteredResource.pSRType.psrType → fuel_type
      - start_DateAndOrTime.date/time                → event_start
      - end_DateAndOrTime.date/time                  → event_end
      - powerSystemResources.nominalP                → installed_capacity_mw
      - quantity (from Period/Point)                 → available_capacity_mw
      - Reason/text                                  → reason_text
    """
    messages = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return messages

    # Detect namespace from root tag
    ns_uri = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    p = f"{{{ns_uri}}}" if ns_uri else ""

    def g(el, tag):
        found = el.find(f".//{p}{tag}")
        return found.text.strip() if found is not None and found.text else None

    for ts in root.findall(f".//{p}TimeSeries"):
        start_date = g(ts, "start_DateAndOrTime.date")
        start_time = (g(ts, "start_DateAndOrTime.time") or "00:00:00Z").replace("Z", "")
        end_date   = g(ts, "end_DateAndOrTime.date")
        end_time   = (g(ts, "end_DateAndOrTime.time") or "23:59:00Z").replace("Z", "")

        if not start_date:
            continue

        nominal_p = g(ts, "production_RegisteredResource.pSRType.powerSystemResources.nominalP")
        quantity   = g(ts, "quantity")

        msg = {
            "doc_type":              doc_type,
            "message_id":            g(ts, "mRID"),
            "event_type":            g(ts, "businessType"),
            "asset_name":            g(ts, "production_RegisteredResource.name") or g(ts, "registeredResource.name"),
            "fuel_type":             g(ts, "production_RegisteredResource.pSRType.psrType"),
            "reason_text":           g(ts, "text"),  # inside <Reason><text>
            "event_start":           f"{start_date}T{start_time}",
            "event_end":             f"{end_date}T{end_time}" if end_date else f"{start_date}T{end_time}",
            "available_capacity_mw": float(quantity) if quantity else None,
            "installed_capacity_mw": float(nominal_p) if nominal_p else None,
        }
        messages.append(msg)

    return messages


def _get_text(element, path, ns):
    """Safely get text from an XML element."""
    el = element.find(path, ns)
    return el.text.strip() if el is not None and el.text else None


# ══════════════════════════════════════════════════════════════════════════════
# 3. JOIN — align REMIT events to hourly price rows
# ══════════════════════════════════════════════════════════════════════════════

def join_prices_and_remit(prices_df: pd.DataFrame, remit_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each hourly price row, attach any active REMIT events.

    Strategy:
      - For each REMIT message, generate the set of hours it covers
      - Group by hour: concatenate reason_text, count active events, sum lost capacity
      - Left-join onto the price series

    Output columns:
      - timestamp_utc, price_eur_mwh                    (from prices)
      - n_active_outages                                 (count of overlapping REMIT events)
      - total_unavailable_mw                             (sum of capacity offline)
      - outage_texts                                     (semicolon-joined reason texts)
      - outage_assets                                    (semicolon-joined asset names)
      - outage_fuel_types                                (semicolon-joined fuel types)
      - has_unplanned_outage                             (bool: any A54 event active)
    """
    print("\n[3/3] Joining prices ← REMIT events...")

    if remit_df.empty:
        prices_df["n_active_outages"] = 0
        prices_df["total_unavailable_mw"] = 0.0
        prices_df["outage_texts"] = ""
        prices_df["outage_assets"] = ""
        prices_df["outage_fuel_types"] = ""
        prices_df["has_unplanned_outage"] = False
        return prices_df

    # Build hourly index for each REMIT event
    hourly_events = []

    for _, row in remit_df.iterrows():
        try:
            evt_start = pd.Timestamp(row["event_start"]).floor("h")
            evt_end = pd.Timestamp(row["event_end"]).ceil("h")
        except Exception:
            continue

        hours = pd.date_range(evt_start, evt_end, freq="h")

        # Calculate unavailable capacity
        installed = row.get("installed_capacity_mw") or 0
        available = row.get("available_capacity_mw") or 0
        unavailable = max(0, installed - available)

        for h in hours:
            hourly_events.append({
                "timestamp_utc": h,
                "reason_text": row.get("reason_text") or "",
                "asset_name": row.get("asset_name") or "",
                "fuel_type": row.get("fuel_type") or "",
                "unavailable_mw": unavailable,
                "is_unplanned": row.get("event_type") == "A54",
            })

    if not hourly_events:
        prices_df["n_active_outages"] = 0
        prices_df["total_unavailable_mw"] = 0.0
        prices_df["outage_texts"] = ""
        prices_df["outage_assets"] = ""
        prices_df["outage_fuel_types"] = ""
        prices_df["has_unplanned_outage"] = False
        return prices_df

    events_df = pd.DataFrame(hourly_events)

    # Aggregate per hour
    agg = events_df.groupby("timestamp_utc").agg(
        n_active_outages=("reason_text", "count"),
        total_unavailable_mw=("unavailable_mw", "sum"),
        outage_texts=("reason_text", lambda x: "; ".join(t for t in x if t)),
        outage_assets=("asset_name", lambda x: "; ".join(t for t in x if t)),
        outage_fuel_types=("fuel_type", lambda x: "; ".join(t for t in x if t)),
        has_unplanned_outage=("is_unplanned", "any"),
    ).reset_index()

    # Left join
    merged = prices_df.merge(agg, on="timestamp_utc", how="left")
    merged["n_active_outages"] = merged["n_active_outages"].fillna(0).astype(int)
    merged["total_unavailable_mw"] = merged["total_unavailable_mw"].fillna(0.0)
    merged["outage_texts"] = merged["outage_texts"].fillna("")
    merged["outage_assets"] = merged["outage_assets"].fillna("")
    merged["outage_fuel_types"] = merged["outage_fuel_types"].fillna("")
    merged["has_unplanned_outage"] = merged["has_unplanned_outage"].fillna(False)

    print(f"  ✓ Joined dataset: {len(merged)} rows")
    print(f"    Hours with ≥1 outage: {(merged.n_active_outages > 0).sum()}")
    print(f"    Hours with unplanned: {merged.has_unplanned_outage.sum()}")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E prices + REMIT data.")
    parser.add_argument("--start",      default=START,      help="Start datetime YYYYMMDDHHMM (default: %(default)s)")
    parser.add_argument("--end",        default=END,        help="End datetime YYYYMMDDHHMM (default: %(default)s)")
    parser.add_argument("--zone",       default=ZONE,       help="Bidding zone EIC code (default: %(default)s)")
    parser.add_argument("--zone-label", default=ZONE_LABEL, help="Short label for output filenames (default: %(default)s)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for output CSVs (default: %(default)s)")
    args = parser.parse_args()

    START      = args.start
    END        = args.end
    ZONE       = args.zone
    ZONE_LABEL = args.zone_label
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if API_TOKEN == "YOUR_TOKEN_HERE":
        print("=" * 70)
        print("SETUP REQUIRED")
        print("=" * 70)
        print()
        print("1. Register at https://transparency.entsoe.eu/")
        print("2. Go to Settings → Web API Security Token")
        print("3. Either:")
        print("     export ENTSOE_API_TOKEN='your-token-here'")
        print("   or edit the API_TOKEN variable at the top of this script")
        print()
        print("Then re-run: python fetch_entsoe.py")
        print("=" * 70)

        # Generate sample output so you can see the schema
        print("\nGenerating sample schema preview...\n")

        sample_prices = pd.DataFrame({
            "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h"),
            "price_eur_mwh": [52.3, 48.1, 45.7, 43.2, 51.8],
        })

        sample_remit = pd.DataFrame({
            "doc_type": ["A78", "A78"],
            "message_id": ["REMIT-001", "REMIT-002"],
            "event_type": ["A53", "A54"],
            "asset_name": ["Kraftwerk Moorburg Block A", "Isar 2"],
            "fuel_type": ["B02", "B14"],
            "reason_text": [
                "Planned maintenance of turbine blade assembly",
                "Unplanned trip due to cooling system fault",
            ],
            "event_start": ["2024-01-01T01:00", "2024-01-01T03:00"],
            "event_end": ["2024-01-01T04:00", "2024-01-01T05:00"],
            "available_capacity_mw": [0, 200],
            "installed_capacity_mw": [800, 1400],
        })

        print("PRICE SERIES SCHEMA:")
        print(sample_prices.to_string(index=False))
        print()
        print("REMIT MESSAGES SCHEMA:")
        print(sample_remit.to_string(index=False))
        print()
        print("JOINED OUTPUT COLUMNS:")
        print("  timestamp_utc, price_eur_mwh, n_active_outages,")
        print("  total_unavailable_mw, outage_texts, outage_assets,")
        print("  outage_fuel_types, has_unplanned_outage")

        sys.exit(0)

    # ── Fetch ──────────────────────────────────────────────────────────────
    prices = fetch_day_ahead_prices(ZONE, START, END)
    remit_path = os.path.join(OUTPUT_DIR, f"entsoe_remit_{ZONE_LABEL}.csv")
    remit = fetch_remit_messages(ZONE, START, END, incremental_path=remit_path)

    # ── Save raw ───────────────────────────────────────────────────────────
    prices_path = os.path.join(OUTPUT_DIR, f"entsoe_prices_{ZONE_LABEL}.csv")

    prices.to_csv(prices_path, index=False)
    print(f"\n  ✓ Saved {prices_path}")
    print(f"  ✓ REMIT saved incrementally to {remit_path}")

    # ── Join ───────────────────────────────────────────────────────────────
    joined = join_prices_and_remit(prices, remit)
    joined_path = os.path.join(OUTPUT_DIR, f"entsoe_joined_{ZONE_LABEL}.csv")
    joined.to_csv(joined_path, index=False)
    print(f"  ✓ Saved {joined_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Prices:  {len(prices)} hourly rows")
    print(f"  REMIT:   {len(remit)} outage messages")
    print(f"  Joined:  {len(joined)} rows ({(joined.n_active_outages > 0).sum()} with active outages)")
    print(f"\nFiles:")
    print(f"  {prices_path}")
    print(f"  {remit_path}")
    print(f"  {joined_path}")
