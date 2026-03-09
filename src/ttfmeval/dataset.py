"""Dataset and data loading for Migas-1.5 evaluation."""

import os
import random
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = (".csv", ".parquet")


def read_datafile(path: str) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame based on file extension.

    Args:
        path: Path to a .csv or .parquet file.

    Returns:
        DataFrame with the file contents.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported file format '{ext}' for {path}. "
        f"Supported formats: {SUPPORTED_EXTENSIONS}"
    )


def get_datasets_dir_from_hf(
    repo_id: str,
    subdir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Download a Hugging Face dataset repo and return the path to its contents.

    Use this path as datasets_dir for evaluation (it will contain CSV files with
    columns t, y_t, text). For private repos, set the HF_TOKEN environment variable
    or pass token=.

    Args:
        repo_id: Hugging Face repo id (e.g. Synthefy/migas-1.5-sample-datasets).
        subdir: Subdirectory inside the repo to use as the root for CSVs. Defaults to None (repo root).
        cache_dir: Directory to cache the download. Defaults to HF hub cache.
        token: Hugging Face token for private repos. Defaults to HF_TOKEN env.

    Returns:
        Path to the directory containing the dataset files (or the specified subdir).
    """
    from huggingface_hub import snapshot_download

    if token is None:
        token = os.environ.get("HF_TOKEN")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        token=token,
    )
    if subdir:
        path = os.path.join(path, subdir)
    return path


def list_data_files(data_dir: str) -> List[str]:
    """Find all CSV and Parquet files in a directory (recursively), skipping derived files.

    Skips files whose basename (without extension) ends with _embeddings, _original,
    _temp, _results, or _temp_results.

    Args:
        data_dir: Root directory to walk for data files.

    Returns:
        Sorted list of full paths to CSV/Parquet files that pass the filter.
    """
    all_files = []
    for root, _dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(SUPPORTED_EXTENSIONS):
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


# Backward-compatible alias
list_csv_files = list_data_files


class LateFusionDataset(Dataset):
    """Dataset for late fusion evaluation with time series and text context.

    Each item is a window of length seq_len + pred_len from a data file (t, y_t, text).
    Supports CSV and Parquet formats. History is normalized per-window (mean/std).
    For test split, windows are enumerated with stride; for train/val, random windows
    are sampled each __getitem__.
    """

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
        """Initialize the dataset.

        Args:
            seq_len: Length of context (history) in each window.
            pred_len: Length of prediction horizon (last pred_len steps are targets).
            datasets: List of paths to data files (CSV or Parquet; columns: t, y_t, text).
            split: "train", "val", or "test". Defaults to "train".
            val_length: Effective length per epoch for train/val (number of __getitem__ calls). Defaults to 1000.
            stride: Step between consecutive test windows. Defaults to 1.
            test_cutoff: Only use rows with t <= this date (YYYYMMDD) for train/val. Defaults to "20221231".
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_files = datasets
        self.epoch_length = 10000 if split == "train" else val_length
        self.split = split
        self.stride = stride
        self.test_cutoff_datetime = pd.to_datetime(test_cutoff, format="%Y%m%d")

        if split == "test":
            self.windows = []
            for data_file in self.data_files:
                df = read_datafile(data_file)
                df["t_date"] = pd.to_datetime(df["t"], errors="coerce")
                df = df[df["t_date"].notna()]
                df = df.drop(columns=["t_date"])
                for i in range(0, len(df) - self.seq_len + 1, self.stride):
                    self.windows.append((data_file, df.iloc[i : i + self.seq_len]))

    def __len__(self) -> int:
        if self.split == "test":
            return len(self.windows)
        return self.epoch_length

    def __getitem__(self, index: int) -> dict:
        if self.split == "test":
            data_file, df = self.windows[index]
        else:
            try:
                data_file = random.choice(self.data_files)
                df = read_datafile(data_file)
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
            "dataset_name": data_file,
            "history_mean": history_mean,
            "history_std": history_std,
            "timestamps": timestamps,
        }


def collate_fn(batch: List[dict]) -> dict:
    """Collate a list of LateFusionDataset items into a batch dict.

    Args:
        batch: List of dicts from __getitem__ (ts, text, index, dataset_name, history_mean, history_std, timestamps).

    Returns:
        Dict with "ts" (stacked tensor), "text" (list of lists), "index", "dataset_name",
        "history_means", "history_stds", "timestamps" (lists).
    """
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
