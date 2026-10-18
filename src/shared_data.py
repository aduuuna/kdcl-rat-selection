"""Generic, dataset-agnostic tabular utilities shared across this project's pipeline. Originally
part of a much larger file written for an earlier, abandoned phase of this project (a different,
unrelated dataset), trimmed 2026-09-23 to just the pieces still genuinely imported by rat_dataset.py,
train.py, evaluate.py, robustness.py, and recommend.py. Nothing here assumes any particular column
names or RAT pair, that's rat_dataset.py's job.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def make_load_label(df: pd.DataFrame, load_col: str, n_classes: int) -> pd.Series:
    """Quantile-based binning (each bin gets roughly equal row count), not fixed-width, since raw
    throughput is heavily skewed."""
    return pd.qcut(df[load_col], q=n_classes, labels=False, duplicates="drop")


def fit_scaler(df: pd.DataFrame, feature_cols: list):
    """Mean/std from training data only -- avoids leaking val/test statistics. Distortion
    magnitudes (src/distortions.py) are calibrated assuming features arrive in this normalized
    space, so this must run before distortions are applied, not after."""
    values = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def time_blocked_split(df: pd.DataFrame, order_cols: tuple, val_frac: float = 0.15, test_frac: float = 0.15):
    """Sequential split, not random shuffle -- wireless data is temporally correlated, so a random
    split would leak future-period information into training. order_cols has no default; every
    caller in this project passes ("time_utc",) explicitly."""
    df_sorted = df.sort_values(list(order_cols)).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    train_df = df_sorted.iloc[: n - n_val - n_test]
    val_df = df_sorted.iloc[n - n_val - n_test: n - n_test]
    test_df = df_sorted.iloc[n - n_test:]
    return train_df, val_df, test_df


class SingleViewDataset(Dataset):
    """No distortion -- used for validation, testing, and inference, where you want the real,
    undistorted signal rather than a training-time-only augmentation."""

    def __init__(self, df: pd.DataFrame, feature_cols: list, label: pd.Series = None, scaler=None):
        features = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        if scaler is not None:
            mean, std = scaler
            features = (features - mean) / std
        self.features = features
        self.labels = label.to_numpy(dtype=np.int64) if label is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = int(self.labels[idx]) if self.labels is not None else -1
        return x, y
