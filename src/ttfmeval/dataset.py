"""Dataset and data loading for TTFM evaluation."""

import os
import random
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def list_csv_files(data_dir: str) -> List[str]:
    """Find all CSV files in directory (recursively), skipping derived files."""
    all_files = []
    for root, _dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    def is_base(p: str) -> bool:
        name = os.path.basename(p)
        base, _ = os.path.splitext(name)
        skip_suffixes = (
            "_embeddings",
            "_original",
            "_temp",
            "_results",
            "_temp_results",
        )
        return not any(base.endswith(s) for s in skip_suffixes)

    return sorted([p for p in all_files if is_base(p)])


class LateFusionDataset(Dataset):
    """Dataset for late fusion evaluation with text context."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        datasets: List[str],
        split: Literal["train", "val", "test"] = "train",
        val_length: int = 1000,
        stride: int = 1,
        test_cutoff: str = "20221231",
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.csvs = datasets
        self.epoch_length = 10000 if split == "train" else val_length
        self.split = split
        self.stride = stride
        self.test_cutoff_datetime = pd.to_datetime(test_cutoff, format="%Y%m%d")

        if split == "test":
            self.windows = []
            for csv in self.csvs:
                df = pd.read_csv(csv)
                df["t_date"] = pd.to_datetime(df["t"], errors="coerce")
                df = df[df["t_date"].notna()]
                df = df.drop(columns=["t_date"])
                for i in range(0, len(df) - self.seq_len + 1, self.stride):
                    self.windows.append((csv, df.iloc[i : i + self.seq_len]))

    def __len__(self) -> int:
        if self.split == "test":
            return len(self.windows)
        return self.epoch_length

    def __getitem__(self, index: int):
        if self.split == "test":
            csv, df = self.windows[index]
        else:
            try:
                csv = random.choice(self.csvs)
                df = pd.read_csv(csv)
            except Exception:
                return self.__getitem__(index + 1)

            df["t_date"] = pd.to_datetime(df["t"], errors="coerce")
            df = df[df["t_date"].notna() & (df["t_date"] <= self.test_cutoff_datetime)]
            df = df.drop(columns=["t_date"])

            if len(df) < self.seq_len:
                return self.__getitem__(index + 1)

            random_index = random.randint(0, len(df) - self.seq_len)
            df = df.iloc[random_index : random_index + self.seq_len]

        df = df.sort_values(by="t")

        if isinstance(df["y_t"].values[0], str):
            df["y_t"] = df["y_t"].apply(lambda x: float(x.replace("$", "")))
        else:
            df["y_t"] = df["y_t"].apply(float)

        ts = df["y_t"].values.astype(np.float32)

        history = ts[: -self.pred_len]
        history_mean = float(history.mean())
        history_std = float(history.std())
        ts = (ts - history_mean) / (history_std + 1e-8)
        ts[np.isnan(ts)] = 0

        text = df["text"].values.tolist()
        timestamps = df["t"].values.tolist()

        return {
            "ts": ts,
            "text": text,
            "index": index,
            "dataset_name": csv,
            "history_mean": history_mean,
            "history_std": history_std,
            "timestamps": timestamps,
        }


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for late fusion dataset."""
    ts_batch = torch.stack(
        [torch.tensor(s["ts"], dtype=torch.float32) for s in batch], dim=0
    )
    return {
        "ts": ts_batch,
        "text": [s["text"] for s in batch],
        "index": [s["index"] for s in batch],
        "dataset_name": [s["dataset_name"] for s in batch],
        "history_means": [s["history_mean"] for s in batch],
        "history_stds": [s["history_std"] for s in batch],
        "timestamps": [s["timestamps"] for s in batch],
    }
