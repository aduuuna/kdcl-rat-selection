"""Data pipeline for the RTR-NetTest Austria dataset (3G vs 4G, 2016-06) -- the "fresh start"
dataset with a real, verified performance gap between RATs (median download: 3G 6,282 kbit vs
4G 33,348 kbit), unlike the original urban dataset where no genuine capacity gap could be found
(see docs/notes/progress_log.md). Reuses the generic pieces from data.py (MultiViewDataset,
SingleViewDataset, fit_scaler, make_load_label, time_blocked_split) -- only the schema-specific
loading/labeling/feature-selection logic is duplicated here.
"""
import numpy as np
import pandas as pd

from data import (  # noqa: F401 -- re-exported for callers
    MultiViewDataset, SingleViewDataset, fit_scaler, make_load_label, time_blocked_split,
)

RAW_CSV = "data/raw/rtr_monthly/netztest-opendata-2016-06.csv"
RATS = ("3G", "4G")

# IDs, device/software metadata, and columns confirmed 100% empty in this file (2026-07-28 EDA) --
# not radio-state features.
NON_FEATURE_COLS = {
    "open_uuid", "open_test_uuid", "time_utc", "cat_technology", "network_type", "loc_src",
    "gkz", "zip_code", "country_location", "server_name", "platform", "model", "client_version",
    "network_mcc_mnc", "network_name", "sim_mcc_mnc", "nat_type", "asn", "ip_anonym", "pinned",
    "kg_nr", "gkz_sa", "cell_area_code", "cell_location_id", "channel_number", "radio_band",
    "link_id", "link_name", "link_distance", "edge_id", "link_frc", "dtm_level", "temperature",
    "signal_classification", "atraster100", "atraster250", "implausible", "test_duration",
    "num_threads",
}

# Missingness is tech-conditional and genuinely informative (e.g. lte_rsrp is 100% missing for
# 2G/3G -- non-LTE phones can't report it -- vs ~36% for 4G), not a data artifact -- kept as
# features with an explicit missing-flag per column, same treatment as the urban dataset's
# neighbor-cell fields.
SPARSE_COLS = ("lte_rsrp", "lte_rsrq", "signal_strength", "land_cover", "settlement_type", "loc_accuracy")

# ndt_* correlate 0.87-0.94 with download/upload_kbit (alternate speed-test methodology measuring
# the same throughput); upload_kbit correlates 0.77 with the label itself -- all excluded to avoid
# predicting the label from near-duplicates of itself (2026-07-28 EDA, same pattern as the urban
# dataset's temp_load/sig_load leak).
LEAKY_COLS = {"ndt_download_kbit", "ndt_upload_kbit", "upload_kbit"}

LOAD_COL = "download_kbit"
LABEL_COL = "load_class"
N_LOAD_CLASSES = 3  # Low / Medium / High downlink throughput, quantile-based


def load_raw(path: str = RAW_CSV, rats: tuple = RATS) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["cat_technology"].isin(rats)].reset_index(drop=True)
    df = df[df["implausible"] == False].reset_index(drop=True)  # noqa: E712 -- 19/37364 rows, data-quality flag
    df["is_" + rats[1]] = (df["cat_technology"] == rats[1]).astype(np.float32)
    for col in SPARSE_COLS:
        df[col + "_missing"] = df[col].isna().astype(np.float32)
    return df


def numeric_feature_columns(df: pd.DataFrame) -> list:
    excluded = NON_FEATURE_COLS | LEAKY_COLS | {LOAD_COL, LABEL_COL}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in excluded]


def split_by_rat(df: pd.DataFrame, rats: tuple = RATS) -> dict:
    """EDA only -- not used for training splits, both branches train on the full dataset."""
    return {rat: df[df["cat_technology"] == rat].reset_index(drop=True) for rat in rats}


if __name__ == "__main__":
    df = load_raw()
    print("feature columns:", len(numeric_feature_columns(df)))
    df[LABEL_COL] = make_load_label(df, load_col=LOAD_COL, n_classes=N_LOAD_CLASSES)
    print("overall", df.shape, "class balance:", df[LABEL_COL].value_counts(normalize=True).to_dict())
    for rat, rat_df in split_by_rat(df).items():
        print(rat, rat_df.shape, "class balance:", rat_df[LABEL_COL].value_counts(normalize=True).to_dict())
