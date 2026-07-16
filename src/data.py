"""Tabular data pipeline: loads the dataset, builds the load-class label, and provides a shared
dataset where both RAT branches see the same row (see MultiViewDataset)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RAW_XLSX = "data/raw/urban_dataset_processed.xlsx"

# ID/timestamp columns -- not radio-state features.
NON_FEATURE_COLS = {
    "Timestamp", "Node", "CellID", "LAC", "NetworkTech", "State", "Test_Status",
    "Mobility", "SessionID", "PSC", "ARFCN", "SecondCell_NODE", "SecondCell_CELLID",
    "SecondCell_PSC", "SecondCell_ARFCN", "NTech1", "NCellid1", "NLAC1", "NCell1",
    "NARFCN1", "Operatorname", "time_seconds",
}

# Correlate 0.79 / 0.66 with derived_load and are almost certainly its components
# (confirmed 2026-07-16 EDA) -- excluded to avoid predicting the label from its own ingredients.
LEAKY_COLS = {"temp_load", "sig_load"}

LOAD_COL = "derived_load"
LABEL_COL = "load_class"
N_LOAD_CLASSES = 3  # Low / Medium / High, quantile-based


def load_raw(path: str = RAW_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df["is_5G"] = (df["NetworkTech"] == "5G").astype(np.float32)
    return df


def make_load_label(df: pd.DataFrame, load_col: str = LOAD_COL, n_classes: int = N_LOAD_CLASSES) -> pd.Series:
    return pd.qcut(df[load_col], q=n_classes, labels=False, duplicates="drop")


def numeric_feature_columns(df: pd.DataFrame) -> list:
    excluded = NON_FEATURE_COLS | LEAKY_COLS | {LOAD_COL, LABEL_COL}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in excluded]


def split_by_rat(df: pd.DataFrame) -> dict:
    """EDA only -- not used for training splits, both branches train on the full dataset."""
    return {rat: df[df["NetworkTech"] == rat].reset_index(drop=True) for rat in ("4G", "5G")}


def time_blocked_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                        order_cols=("SessionID", "ElapsedTime")):
    """Sequential split, not random shuffle -- wireless data is temporally correlated."""
    df_sorted = df.sort_values(list(order_cols)).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    train_df = df_sorted.iloc[: n - n_val - n_test]
    val_df = df_sorted.iloc[n - n_val - n_test: n - n_test]
    test_df = df_sorted.iloc[n - n_test:]
    return train_df, val_df, test_df


class MultiViewDataset(Dataset):
    """Each branch gets the same row through its own distortion, so one shared teacher logit is valid."""

    def __init__(self, df: pd.DataFrame, feature_cols: list, label: pd.Series, distortions: list):
        self.features = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        self.labels = label.to_numpy(dtype=np.int64)
        self.distortions = distortions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        base = self.features[idx]
        views = np.stack([d(base.copy()) for d in self.distortions])
        return torch.from_numpy(views), int(self.labels[idx])


class SingleViewDataset(Dataset):
    """No distortion -- used for validation/inference."""

    def __init__(self, df: pd.DataFrame, feature_cols: list, label: pd.Series = None):
        self.features = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        self.labels = label.to_numpy(dtype=np.int64) if label is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = int(self.labels[idx]) if self.labels is not None else -1
        return x, y


if __name__ == "__main__":
    df = load_raw()
    print("feature columns:", len(numeric_feature_columns(df)))
    df[LABEL_COL] = make_load_label(df)
    print("overall", df.shape, "class balance:", df[LABEL_COL].value_counts(normalize=True).to_dict())
    for rat, rat_df in split_by_rat(df).items():
        print(rat, rat_df.shape, "class balance:", rat_df[LABEL_COL].value_counts(normalize=True).to_dict())
