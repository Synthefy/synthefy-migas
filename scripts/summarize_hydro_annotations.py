#!/usr/bin/env python3
"""
Summarize verbose hydro outage text annotations into concise descriptions.

For each row in the hydropower CSVs, parses the JSON outage events to extract
structured statistics, then uses gpt-oss-120b to produce a concise natural
language summary.

Usage:
    python scripts/summarize_hydro_annotations.py
    python scripts/summarize_hydro_annotations.py --input_dir /data/ttfm_review/hydropower_csvs \
        --output_dir /data/ttfm_review/hydropower_summarized_csvs
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

INPUT_DIR = "/data/ttfm_review/hydropower_csvs"
OUTPUT_DIR = "/data/ttfm_review/hydropower_summarized_csvs"
LLM_BASE_URL = "http://localhost:8004/v1"
LLM_MODEL = "openai/gpt-oss-120b"
MAX_CONCURRENT = 64
MAX_TOKENS = 1024


def parse_outage_events(raw_text: str) -> list[dict]:
    """Extract individual outage event dicts from a row's text annotation."""
    if not raw_text or pd.isna(raw_text) or str(raw_text).strip() == "":
        return []

    raw_text = str(raw_text)
    events = []
    for match in re.finditer(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*(\[.*?\])(?=\n\[|\Z)', raw_text, re.DOTALL):
        timestamp = match.group(1)
        json_str = match.group(2)
        try:
            items = json.loads(json_str)
            for item in items:
                item["event_timestamp"] = timestamp
                events.append(item)
        except json.JSONDecodeError:
            continue
    return events


def aggregate_events(events: list[dict]) -> dict:
    """Compute aggregate statistics from parsed outage events."""
    if not events:
        return {"n_events": 0}

    unique_events = {}
    for e in events:
        key = (
            e.get("asset_name", ""),
            e.get("unavailability_type", ""),
            e.get("reason_text", ""),
            e.get("unavailable_mw", 0),
        )
        if key not in unique_events:
            unique_events[key] = e

    deduped = list(unique_events.values())

    planned = [e for e in deduped if e.get("unavailability_type") == "Planned"]
    unplanned = [e for e in deduped if e.get("unavailability_type") == "Unplanned"]
    other = [e for e in deduped if e.get("unavailability_type") not in ("Planned", "Unplanned")]

    total_unavail_mw = sum(e.get("unavailable_mw", 0) for e in deduped)
    total_installed_mw = sum(e.get("installed_capacity_mw", 0) for e in deduped)

    def summarize_group(group: list[dict]) -> list[str]:
        items = []
        for e in group:
            asset = e.get("asset_name", "Unknown")
            unavail = e.get("unavailable_mw", "?")
            installed = e.get("installed_capacity_mw", "?")
            reason = e.get("reason_text", "").strip()
            remarks = e.get("remarks", "").strip()
            desc = f"{asset} ({unavail}/{installed} MW)"
            if reason:
                desc += f": {reason[:120]}"
            if remarks and remarks != reason:
                desc += f" [{remarks[:80]}]"
            items.append(desc)
        return items

    return {
        "n_events": len(deduped),
        "n_planned": len(planned),
        "n_unplanned": len(unplanned),
        "n_other": len(other),
        "total_unavailable_mw": total_unavail_mw,
        "total_installed_mw": total_installed_mw,
        "planned_details": summarize_group(planned),
        "unplanned_details": summarize_group(unplanned),
        "_planned_events": planned,
        "_unplanned_events": unplanned,
    }


def build_llm_input(date: str, generation_mw: float, agg: dict) -> str:
    """Build a structured prompt for the LLM from aggregated outage stats."""
    if agg["n_events"] == 0:
        return ""

    lines = [
        f"Date: {date}",
        f"Generation: {generation_mw:.1f} MW",
        f"Total outages: {agg['n_events']} ({agg['n_planned']} planned, {agg['n_unplanned']} unplanned)",
        f"Total unavailable capacity: {agg['total_unavailable_mw']} MW (of {agg['total_installed_mw']} MW installed)",
        "",
    ]

    planned_unavail = sum(
        e.get("unavailable_mw", 0) for e in agg.get("_planned_events", [])
    ) if "_planned_events" in agg else 0
    unplanned_unavail = sum(
        e.get("unavailable_mw", 0) for e in agg.get("_unplanned_events", [])
    ) if "_unplanned_events" in agg else 0

    lines.append(f"=== PLANNED OUTAGES ({agg['n_planned']}, {planned_unavail} MW unavailable) ===")
    if agg["planned_details"]:
        for d in agg["planned_details"][:15]:
            lines.append(f"  - {d}")
        if len(agg["planned_details"]) > 15:
            lines.append(f"  ... and {len(agg['planned_details']) - 15} more")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"=== UNPLANNED OUTAGES ({agg['n_unplanned']}, {unplanned_unavail} MW unavailable) ===")
    if agg["unplanned_details"]:
        for d in agg["unplanned_details"][:15]:
            lines.append(f"  - {d}")
        if len(agg["unplanned_details"]) > 15:
            lines.append(f"  ... and {len(agg['unplanned_details']) - 15} more")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


SYSTEM_PROMPT = """You summarize hydropower outage reports into concise, information-dense annotations for a time series dataset. Given structured outage data for a single day, produce exactly TWO paragraphs:

PLANNED: Summarize planned/scheduled outages — count, total unavailable MW, key assets, maintenance reasons, and expected duration if mentioned. (1-2 sentences)

UNPLANNED: Summarize unplanned/forced outages — count, total unavailable MW, key assets, failure reasons (equipment failures, vibrations, leaks, etc.), and severity indicators. (1-2 sentences)

If a category has no outages, write "None." for that paragraph.

Be factual and concise. Use abbreviations (MW, planned/unpl.). Do NOT include the date or generation value — those are in separate columns.

Format:
PLANNED: [summary]
UNPLANNED: [summary]"""


async def summarize_one(
    client: AsyncOpenAI,
    llm_input: str,
    semaphore: asyncio.Semaphore,
    model: str = LLM_MODEL,
) -> str:
    if not llm_input:
        return ""
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": llm_input},
                    ],
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                err = str(e).lower()
                if ("connection" in err or "timeout" in err) and attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                print(f"  LLM error: {e}")
                return f"[Error: {e}]"
    return "[Error: exhausted retries]"


async def process_csv(csv_path: Path, output_path: Path, base_url: str, model: str) -> None:
    print(f"\nProcessing {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    llm_inputs = []
    for _, row in df.iterrows():
        events = parse_outage_events(row.get("text", ""))
        agg = aggregate_events(events)
        llm_input = build_llm_input(str(row["t"]), float(row["y_t"]) if pd.notna(row["y_t"]) else 0.0, agg)
        llm_inputs.append(llm_input)

    non_empty = sum(1 for x in llm_inputs if x)
    print(f"  {non_empty} rows with outage text, {len(df) - non_empty} empty")

    BATCH_SIZE = 128
    summaries = [""] * len(llm_inputs)
    pending = [(i, inp) for i, inp in enumerate(llm_inputs) if inp]

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start : batch_start + BATCH_SIZE]
        print(f"  Summarizing batch {batch_start // BATCH_SIZE + 1}/{(len(pending) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} rows)...")
        tasks = [summarize_one(client, inp, semaphore, model) for _, inp in batch]
        results = await asyncio.gather(*tasks)
        for (idx, _), result in zip(batch, results):
            summaries[idx] = result

    df_out = df.copy()
    df_out["text"] = summaries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    n_success = sum(1 for s in summaries if s and not s.startswith("[Error"))
    n_error = sum(1 for s in summaries if s.startswith("[Error"))
    print(f"  Results: {n_success} summarized, {n_error} errors, {len(df) - non_empty} empty (no outages)")


async def main_async(input_dir: Path, output_dir: Path, base_url: str, model: str) -> None:
    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {input_dir}")
        return

    print(f"Found {len(csvs)} CSVs in {input_dir}")
    for csv_path in csvs:
        output_path = output_dir / csv_path.name
        await process_csv(csv_path, output_path, base_url, model)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Summarize hydro outage annotations")
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--llm_base_url", type=str, default=LLM_BASE_URL)
    parser.add_argument("--llm_model", type=str, default=LLM_MODEL)
    args = parser.parse_args()

    asyncio.run(main_async(Path(args.input_dir), Path(args.output_dir), args.llm_base_url, args.llm_model))


if __name__ == "__main__":
    main()
