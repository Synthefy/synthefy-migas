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
from migaseval.model.util import ContextSummarizer

SEQ_LEN = 512
PRED_LEN = 16
BATCH_SIZE = 128
LLM_BASE_URL = "http://localhost:8004/v1"
LLM_MODEL = "openai/gpt-oss-120b"
SUMMARIZER_CONTEXT_LEN = 32

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
    "hydropower":{
        "csvs_dir": "/data/ttfm_review/hydropower_csvs",
        "summaries_dir": "/data/ttfm_review/hydropower",
    }
}


def extract_dataset_name(csv_path: str) -> str:
    filename = os.path.basename(csv_path)
    return os.path.splitext(filename)[0]


def store_summaries_for_dataset(
    csv_path: str,
    summaries_save_dir: str,
    context_summarizer: ContextSummarizer,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
    batch_size: int = BATCH_SIZE,
    summarizer_context_len: int = SUMMARIZER_CONTEXT_LEN,
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

    added_counts = {
        "historic_values": 0, "forecast_values": 0,
        "history_mean": 0, "history_std": 0, "summary": 0,
    }

    pending_summary_indices = []
    pending_summary_text_inputs = []
    pending_summary_values_inputs = []

    all_sample_data = []
    all_sample_paths = []

    sample_idx = 0
    for batch_dict in tqdm(loader, desc=f"  Loading {dataset_name}"):
        batch_size_actual = batch_dict["ts"].shape[0]
        batch_text = batch_dict["text"]

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

            if "history_mean" not in data:
                data["history_mean"] = float(batch_dict["history_means"][idx])
                added_counts["history_mean"] += 1

            if "history_std" not in data:
                data["history_std"] = float(batch_dict["history_stds"][idx])
                added_counts["history_std"] += 1

            if "summary" not in data:
                text_per_element = batch_text[idx]
                historic_text = text_per_element[: len(historic)]
                trimmed_text = historic_text[-summarizer_context_len:]
                trimmed_values = historic[-summarizer_context_len:].tolist()
                pending_summary_indices.append(len(all_sample_data))
                pending_summary_text_inputs.append(trimmed_text)
                pending_summary_values_inputs.append(trimmed_values)

            all_sample_data.append(data)
            all_sample_paths.append(summary_path)
            sample_idx += 1

    if pending_summary_indices:
        print(f"  Generating {len(pending_summary_indices)} missing summaries via LLM...")
        for chunk_start in range(0, len(pending_summary_indices), LLM_BATCH_SIZE):
            chunk_end = min(chunk_start + LLM_BATCH_SIZE, len(pending_summary_indices))
            summaries = context_summarizer.summarize_batch(
                pending_summary_text_inputs[chunk_start:chunk_end],
                pending_summary_values_inputs[chunk_start:chunk_end],
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
    summaries_dir = preset["summaries_dir"]

    print(f"\n{'=' * 80}")
    print(f"  Dataset: {preset_name}")
    print(f"  CSVs dir: {csvs_dir}")
    print(f"  Summaries dir: {summaries_dir}")
    print(f"{'=' * 80}\n")

    dataset_csvs = list_csv_files(csvs_dir)
    if not dataset_csvs:
        print(f"  No CSVs found in {csvs_dir}, skipping.")
        return

    print(f"Found {len(dataset_csvs)} CSV files:")
    for p in dataset_csvs:
        print(f"  - {os.path.basename(p)}")

    os.makedirs(summaries_dir, exist_ok=True)

    context_summarizer = ContextSummarizer(
        base_url=args.llm_base_url,
        model_name=args.llm_model,
        max_concurrent=128,
        max_tokens=512,
    )

    for csv_path in tqdm(dataset_csvs, desc=f"Datasets ({preset_name})"):
        store_summaries_for_dataset(
            csv_path=csv_path,
            summaries_save_dir=summaries_dir,
            context_summarizer=context_summarizer,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
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
