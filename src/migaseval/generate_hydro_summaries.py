#!/usr/bin/env python3
"""
Generate and cache LLM summaries for all datasets.

Hardcoded paths for fnspid, suite, and trading datasets.

Usage:
    python -m migaseval.generate_summaries --dataset fnspid
    python -m migaseval.generate_summaries --dataset trading
    python -m migaseval.generate_summaries --dataset suite
    python -m migaseval.generate_summaries --dataset all
"""

import os
import json
import argparse

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from migaseval.dataset import LateFusionDataset, collate_fn as late_fusion_collate, list_data_files as list_csv_files
from migaseval.model.util import HydroContextSummarizer

SEQ_LEN = 512
PRED_LEN = 16
BATCH_SIZE = 128
LLM_BASE_URL = "http://localhost:8004/v1"
LLM_MODEL = "openai/gpt-oss-120b"
SUMMARIZER_CONTEXT_LEN = 32

FORECAST_MODES = ("none", "planned", "all")

FORECAST_MODE_DIR_SUFFIX = {
    "none": "_no_forecast",
    "planned": "_planned_forecast",
    "all": "_all_forecast",
}

DATASET_PRESETS = {
    "fnspid": {
        "csvs_dir": "/data/ttfm_review/fnspid_0.5_complement_csvs",
        "summaries_dir": "/data/ttfm_review/fnspid_0.5_complement",
    },
    "trading": {
        "csvs_dir": "/data/ttfm_review/trading_economics_refined_csvs",
        "summaries_dir": "/data/ttfm_review/trading_economics_refined",
    },
    "suite": {
        "csvs_dir": "/data/ttfm_review/icml_suite_csvs",
        "summaries_dir": "/data/ttfm_review/icml_suite",
    },
    "hydropower": {
        "csvs_dir": "/data/ttfm_review/hydropower_summarized_csvs",
        "summaries_dir": "/data/ttfm_review/hydropower_full_stride",
    },
}

# Zone-specific domain context describing the empirical relationship between
# outage text annotations and hydro reservoir generation capacity.
ZONE_META_CONTEXT = {
    "NO_1": (
        "This is Norwegian hydro bidding zone NO_1 (~200–2000 MW generation range). "
        "There is a strong negative correlation between outage count and generation: "
        "as outage count increases from 0 to 400, generation drops from ~1500 MW to ~200–500 MW. "
        "Unplanned outages cluster at lower generation values more than planned ones. "
        "Lagged impact analysis shows high-outage days cause generation to drop ~1.5% by day 3–5, "
        "with persistent depression through day 14. Outage events are bursty — concentrated in "
        "certain periods rather than continuously present."
    ),
    "NO_2": (
        "This is Norwegian hydro bidding zone NO_2, the largest zone (~3000–8000 MW). "
        "Outage counts are persistently high (100–500/day). Generation dips often align with "
        "outage spikes, with a moderate negative trend visible but noisier than NO_1. "
        "At low outage counts (<200), generation spans the full range; at high counts (>1000), "
        "generation compresses to ~2000–5000 MW. Lagged impact shows a sharp ~4% dip peaking "
        "around day 3–4 after high-outage days, followed by partial recovery — the large fleet "
        "can compensate over time. Summer months tend to have lower outage counts and higher "
        "generation; spring snowmelt brings high generation peaks even with outages present."
    ),
    "SE_1": (
        "This is Swedish hydro bidding zone SE_1 (~500–4500 MW). "
        "Both outages and generation show strong seasonal patterns. Outages ramp up in "
        "spring/summer, which is also when generation peaks due to snowmelt inflows. "
        "This creates a confounding effect: planned maintenance is strategically scheduled "
        "during high-inflow periods when spare capacity exists, so outages and generation can "
        "move together seasonally. Unplanned outages (high unplanned ratio) cluster at extreme "
        "low-generation values, indicating a stronger association with genuine capacity loss. "
        "The scatter shows a clear fan from high-generation/low-outages to low-generation/high-outages."
    ),
    "SE_3": (
        "This is Swedish hydro bidding zone SE_3 (~400–1900 MW), a smaller zone with fewer "
        "outages overall. Outage patterns show 'block' structures — fixed numbers of outages "
        "persisting for weeks, suggesting long-duration planned maintenance rather than sporadic "
        "events. The outage-generation relationship is weak and noisy; most outage counts cluster "
        "below 50, making clean conclusions difficult. Lagged analysis shows an inverted pattern: "
        "zero-outage days exhibit the strongest positive generation trend (+2% over 14 days), "
        "while low-outage days decline ~3% by day 14. Outages here may be compensated by other "
        "units, and low-outage periods can coincide with seasonal generation decline."
    ),
}

GENERAL_HYDRO_META_CONTEXT = (
    "This is a hydropower reservoir generation time series. The text annotations describe "
    "power plant outage events (planned and unplanned) affecting generation capacity. "
    "Key domain insights: (1) Unplanned outages correlate more strongly with generation loss "
    "than planned outages, since planned maintenance is scheduled when spare capacity exists. "
    "(2) Outage impacts on generation are often lagged by 3–5 days. "
    "(3) Seasonal patterns matter — spring snowmelt brings high inflows and generation peaks, "
    "while planned maintenance is often scheduled during high-inflow periods. "
    "(4) The total unavailable MW from concurrent outages is more informative than raw outage count."
)


def filter_planned_outages(text: str) -> str:
    """Extract only the PLANNED paragraph from a summarized outage annotation.

    The summarized CSVs have text in the format:
        PLANNED: [summary]
        UNPLANNED: [summary]
    This returns only the PLANNED line content, stripping out UNPLANNED.
    Also handles raw JSON text as a fallback.
    """
    if not text or str(text).strip() == "" or text != text:
        return ""

    text = str(text).strip()

    import re as _re
    match = _re.search(r'PLANNED:\s*(.+?)(?=\nUNPLANNED:|\Z)', text, _re.DOTALL)
    if match:
        planned = match.group(1).strip()
        if planned.lower() == "none." or not planned:
            return ""
        return f"PLANNED: {planned}"

    return ""


def extract_dataset_name(csv_path: str) -> str:
    filename = os.path.basename(csv_path)
    return os.path.splitext(filename)[0]


def extract_zone(csv_path: str) -> str | None:
    """Extract the hydro zone identifier (e.g. NO_1, SE_3) from the CSV filename."""
    name = extract_dataset_name(csv_path).upper()
    for zone in ZONE_META_CONTEXT:
        if name.startswith(zone):
            return zone
    return None


def get_meta_context(csv_path: str) -> str:
    """Return zone-specific meta context if the zone is recognized, otherwise general context."""
    zone = extract_zone(csv_path)
    if zone and zone in ZONE_META_CONTEXT:
        return ZONE_META_CONTEXT[zone]
    return GENERAL_HYDRO_META_CONTEXT


def store_summaries_for_dataset(
    csv_path: str,
    summaries_save_dir: str,
    context_summarizer: HydroContextSummarizer,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
    batch_size: int = BATCH_SIZE,
    summarizer_context_len: int = SUMMARIZER_CONTEXT_LEN,
    forecast_mode: str = "planned",
):
    """Generate and store LLM summaries for a single CSV dataset.

    For each sample, only missing fields are added. If a field already exists,
    consistency is verified against the DataLoader output.
    """
    LLM_BATCH_SIZE = 128
    dataset_name = extract_dataset_name(csv_path)
    save_dir = os.path.join(summaries_save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    dataset = LateFusionDataset(
        seq_len + pred_len,
        pred_len,
        [csv_path],
        split="test",
        val_length=1000,
        stride=1,
    )
    if len(dataset) == 0:
        print(f"  {dataset_name}: empty dataset, skipping.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=late_fusion_collate,
    )

    meta_context = get_meta_context(csv_path)

    added_counts = {
        "historic_values": 0, "forecast_values": 0,
        "historic_timestamps": 0, "forecast_timestamps": 0,
        "history_mean": 0, "history_std": 0, "summary": 0,
    }

    pending_summary_indices = []
    pending_summary_text_inputs = []
    pending_summary_values_inputs = []
    pending_summary_timestamps_inputs = []
    pending_summary_forecast_text = []
    pending_summary_forecast_timestamps = []

    all_sample_data = []
    all_sample_paths = []

    sample_idx = 0
    for batch_dict in tqdm(loader, desc=f"  Loading {dataset_name}"):
        batch_size_actual = batch_dict["ts"].shape[0]
        batch_text = batch_dict["text"]
        batch_timestamps = batch_dict["timestamps"]

        for idx in range(batch_size_actual):
            summary_path = os.path.join(save_dir, f"summary_{sample_idx}.json")

            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            values = batch_dict["ts"][idx].cpu().numpy()
            historic = values[:-pred_len]
            forecast = values[-pred_len:]

            if "historic_values" in data:
                stored = np.array(data["historic_values"], dtype=np.float32)
                assert np.allclose(stored, historic, atol=1e-5), \
                    f"sample {sample_idx}: stored historic_values mismatch"
            else:
                data["historic_values"] = historic.tolist()
                added_counts["historic_values"] += 1

            if "forecast_values" in data:
                stored = np.array(data["forecast_values"], dtype=np.float32)
                assert np.allclose(stored, forecast, atol=1e-5), \
                    f"sample {sample_idx}: stored forecast_values mismatch"
            else:
                data["forecast_values"] = forecast.tolist()
                added_counts["forecast_values"] += 1

            timestamps_per_element = batch_timestamps[idx]
            hist_ts = timestamps_per_element[: len(historic)]
            fcast_ts = timestamps_per_element[len(historic):]

            if "historic_timestamps" not in data:
                data["historic_timestamps"] = list(hist_ts)
                added_counts["historic_timestamps"] += 1

            if "forecast_timestamps" not in data:
                data["forecast_timestamps"] = list(fcast_ts)
                added_counts["forecast_timestamps"] += 1

            if "history_mean" not in data:
                data["history_mean"] = float(batch_dict["history_means"][idx])
                added_counts["history_mean"] += 1

            if "history_std" not in data:
                data["history_std"] = float(batch_dict["history_stds"][idx])
                added_counts["history_std"] += 1

            if "summary" not in data:
                text_per_element = batch_text[idx]
                historic_text = text_per_element[: len(historic)]
                historic_timestamps = hist_ts
                trimmed_text = historic_text[-summarizer_context_len:]
                trimmed_values = historic[-summarizer_context_len:].tolist()
                trimmed_timestamps = historic_timestamps[-summarizer_context_len:]

                if forecast_mode == "none":
                    forecast_text_window = None
                    forecast_timestamps_window = None
                else:
                    forecast_text_raw = text_per_element[len(historic):]
                    forecast_timestamps_window = list(fcast_ts)
                    if forecast_mode == "planned":
                        forecast_text_window = [
                            filter_planned_outages(t) for t in forecast_text_raw
                        ]
                    else:
                        forecast_text_window = list(forecast_text_raw)

                pending_summary_indices.append(len(all_sample_data))
                pending_summary_text_inputs.append(trimmed_text)
                pending_summary_values_inputs.append(trimmed_values)
                pending_summary_timestamps_inputs.append(trimmed_timestamps)
                pending_summary_forecast_text.append(forecast_text_window)
                pending_summary_forecast_timestamps.append(forecast_timestamps_window)

            all_sample_data.append(data)
            all_sample_paths.append(summary_path)
            sample_idx += 1

    if pending_summary_indices:
        print(f"  Generating {len(pending_summary_indices)} missing summaries via LLM...")
        print(f"  Using meta context for: {extract_zone(csv_path) or 'general'}")
        for chunk_start in range(0, len(pending_summary_indices), LLM_BATCH_SIZE):
            chunk_end = min(chunk_start + LLM_BATCH_SIZE, len(pending_summary_indices))
            summaries = context_summarizer.summarize_batch(
                pending_summary_text_inputs[chunk_start:chunk_end],
                pending_summary_values_inputs[chunk_start:chunk_end],
                pending_summary_timestamps_inputs[chunk_start:chunk_end],
                meta_context=meta_context,
                forecast_text_batch=pending_summary_forecast_text[chunk_start:chunk_end],
                forecast_timestamps_batch=pending_summary_forecast_timestamps[chunk_start:chunk_end],
            )
            for i, summary in enumerate(summaries):
                pos = pending_summary_indices[chunk_start + i]
                all_sample_data[pos]["summary"] = summary
                added_counts["summary"] += 1

    for path, data in zip(all_sample_paths, all_sample_data):
        with open(path, "w") as f:
            json.dump(data, f)

    print(f"  {dataset_name}: {sample_idx} samples in {save_dir}")
    for field, count in added_counts.items():
        if count > 0:
            print(f"    added {field}: {count}")


def run_for_preset(preset_name: str, preset: dict, args):
    csvs_dir = preset["csvs_dir"]
    base_summaries_dir = preset["summaries_dir"]

    forecast_mode = getattr(args, "forecast_mode", "planned")

    if preset_name == "hydropower":
        summaries_dir = base_summaries_dir + FORECAST_MODE_DIR_SUFFIX[forecast_mode]
    else:
        summaries_dir = base_summaries_dir

    print(f"\n{'=' * 80}")
    print(f"  Dataset: {preset_name}")
    print(f"  CSVs dir: {csvs_dir}")
    print(f"  Summaries dir: {summaries_dir}")
    if preset_name == "hydropower":
        print(f"  Forecast mode: {forecast_mode}")
    print(f"{'=' * 80}\n")

    dataset_csvs = list_csv_files(csvs_dir)
    if not dataset_csvs:
        print(f"  No CSVs found in {csvs_dir}, skipping.")
        return

    print(f"Found {len(dataset_csvs)} CSV files:")
    for p in dataset_csvs:
        print(f"  - {os.path.basename(p)}")

    os.makedirs(summaries_dir, exist_ok=True)

    context_summarizer = HydroContextSummarizer(
        base_url=args.llm_base_url,
        model_name=args.llm_model,
        max_concurrent=128,
        max_tokens=1024,
    )

    for csv_path in tqdm(dataset_csvs, desc=f"Datasets ({preset_name})"):
        store_summaries_for_dataset(
            csv_path=csv_path,
            summaries_save_dir=summaries_dir,
            context_summarizer=context_summarizer,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            forecast_mode=forecast_mode if preset_name == "hydropower" else "none",
        )

    print(f"\nDone: {preset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and cache LLM summaries for evaluation datasets",
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=list(DATASET_PRESETS.keys()) + ["all"],
        help="Which dataset collection to generate summaries for",
    )
    parser.add_argument(
        "--forecast_mode",
        default="planned",
        choices=list(FORECAST_MODES),
        help=(
            "How to handle forecast-period text for hydropower summaries. "
            "'none': no forecast text, "
            "'planned': only planned outages, "
            "'all': both planned and unplanned outages. "
            "Each mode stores summaries in a separate directory."
        ),
    )
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--llm_base_url", type=str, default=LLM_BASE_URL)
    parser.add_argument("--llm_model", type=str, default=LLM_MODEL)
    args = parser.parse_args()

    if args.dataset == "all":
        targets = DATASET_PRESETS
    else:
        targets = {args.dataset: DATASET_PRESETS[args.dataset]}

    for name, preset in targets.items():
        run_for_preset(name, preset, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
