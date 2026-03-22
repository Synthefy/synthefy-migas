"""
Use Claude to classify UMM reason texts into semantic buckets.

Sends batches of unique reason_text values to Claude, gets back
bucket assignments, saves as JSON mapping.

Usage:
  uv run python scripts/classify_buckets_llm.py
"""

import glob
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import anthropic

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

DATA_DIR = os.path.join(_ROOT_DIR, "data/entsoe_examples_new")
OUTPUT_PATH = os.path.join(_ROOT_DIR, "data/reason_text_buckets.json")

# ── Load all unique reason texts with counts ──────────────────────────────

frames = []
for folder in sorted(glob.glob(f"{DATA_DIR}/*")):
    csv = glob.glob(f"{folder}/umm_parsed_*.csv")
    if not csv:
        continue
    df = pd.read_csv(csv[0])
    frames.append(df)

umm = pd.concat(frames)
counts = umm.reason_text.dropna().value_counts()

# Focus on texts with 2+ occurrences first (2,179 texts, covers ~85% of events)
# Then we'll handle rare ones with a fallback
frequent = counts[counts >= 2].index.tolist()
rare = counts[counts == 1].index.tolist()

print(f"Total unique: {len(counts)}")
print(f"Frequent (2+): {len(frequent)} texts covering {counts[counts >= 2].sum()} events")
print(f"Rare (1x): {len(rare)} texts covering {len(rare)} events")

# ── Load existing mapping if resuming ─────────────────────────────────────

if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        mapping = json.load(f)
    print(f"Loaded existing mapping: {len(mapping)} entries")
else:
    mapping = {}

# Filter out already-classified texts
to_classify = [t for t in frequent if t not in mapping]
print(f"Still need to classify: {len(to_classify)} frequent texts")

# ── Classify in batches ───────────────────────────────────────────────────

BATCH_SIZE = 200
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are classifying power plant outage descriptions from European energy markets (REMIT UMM messages).

Given a list of outage reason texts, assign each one to exactly one semantic bucket.

Use these bucket names (pick the most specific match):

EQUIPMENT FAILURE:
- "Turbine failure" — turbine mechanical issues, trips, vibrations
- "Boiler failure" — boiler, steam, furnace issues
- "Generator failure" — generator, stator, rotor issues
- "Transformer failure" — transformer faults, trips
- "Cooling system failure" — cooling, condenser issues
- "Fuel system failure" — fuel feeding, coal mill, gas supply issues
- "Control system failure" — control systems, electronics, automation
- "Electrical failure" — electrical faults, short circuits, switchgear

ENVIRONMENTAL:
- "Icing/cold weather" — icing on wind turbines, cold temperature effects
- "Weather event" — storms, flooding, lightning (not icing)
- "Environmental compliance" — emissions, pollution, fish protection, oil leaks

GRID:
- "Grid outage" — external grid outage, TSO maintenance, transmission issues
- "Grid constraint" — overload, capacity limitation, system protection

MAINTENANCE:
- "Planned maintenance" — scheduled maintenance, annual overhaul, inspection
- "Unplanned repair" — emergency repair, fixing a specific failure
- "Testing/commissioning" — tests, trial runs, commissioning new units

OPERATIONAL:
- "Start-up issue" — start failure, ramp problems, not ready to start
- "Reduced output" — partial reduction, derated operation, load limitation
- "Fuel shortage" — fuel supply issues, no fuel available

OTHER:
- "Nuclear specific" — nuclear scram, reactor issues, nuclear safety
- "New unit" — new production unit, first commissioning
- "Unknown/unspecified" — unknown, awaiting information, no reason given
- "Other" — anything that doesn't fit above

Return ONLY a JSON object mapping each input text to its bucket name. No explanation."""


def classify_batch(texts: list[str]) -> dict:
    """Send a batch of texts to Claude, get back bucket mapping."""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Classify these {len(texts)} outage reason texts into buckets:\n\n{numbered}"
        }],
    )

    # Parse JSON from response
    text = response.content[0].text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    result = json.loads(text)

    # Map back from numbered keys or direct text keys
    out = {}
    for key, bucket in result.items():
        # Key might be "1" or "1. Maintenance" or the text itself
        if key in texts:
            out[key] = bucket
        else:
            # Try to match by number
            try:
                idx = int(str(key).split(".")[0]) - 1
                if 0 <= idx < len(texts):
                    out[texts[idx]] = bucket
            except (ValueError, IndexError):
                pass

    return out


# Process in batches
n_batches = (len(to_classify) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\nProcessing {len(to_classify)} texts in {n_batches} batches of {BATCH_SIZE}...")

for i in range(0, len(to_classify), BATCH_SIZE):
    batch = to_classify[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1

    print(f"\n  Batch {batch_num}/{n_batches} ({len(batch)} texts)...")

    try:
        result = classify_batch(batch)
        mapping.update(result)
        print(f"    ✓ Classified {len(result)} texts")

        # Save incrementally
        with open(OUTPUT_PATH, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"    ✗ Error: {e}")
        # Save what we have so far
        with open(OUTPUT_PATH, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    time.sleep(1)  # rate limit courtesy

# ── Handle rare texts: classify in larger batches ─────────────────────────

to_classify_rare = [t for t in rare if t not in mapping]
print(f"\nClassifying {len(to_classify_rare)} rare texts...")

RARE_BATCH_SIZE = 300
n_rare_batches = (len(to_classify_rare) + RARE_BATCH_SIZE - 1) // RARE_BATCH_SIZE

for i in range(0, len(to_classify_rare), RARE_BATCH_SIZE):
    batch = to_classify_rare[i:i + RARE_BATCH_SIZE]
    batch_num = i // RARE_BATCH_SIZE + 1

    print(f"\n  Rare batch {batch_num}/{n_rare_batches} ({len(batch)} texts)...")

    try:
        result = classify_batch(batch)
        mapping.update(result)
        print(f"    ✓ Classified {len(result)} texts")

        with open(OUTPUT_PATH, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"    ✗ Error: {e}")
        with open(OUTPUT_PATH, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    time.sleep(1)

# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"DONE")
print(f"{'=' * 60}")
print(f"Total classified: {len(mapping)}")
print(f"Saved to: {OUTPUT_PATH}")

# Show bucket distribution
bucket_counts = pd.Series(mapping).value_counts()
print(f"\nBucket distribution:")
for bucket, count in bucket_counts.items():
    print(f"  {count:5d}x  {bucket}")
