"""Data pipeline for the RTR-NetzTest Austria dataset. PIVOTED 2026-09-23 from the original 3G vs
4G / 2016-06 slice to 4G vs 5G / 2024-2025: the merged 2012-2026 dump (data/netztest-opendata-ALL-
2012-2026.csv, 12.6GB, never committed) was filtered via src/extract_4g5g.py (4G+5G, plausible
rows, proper CSV-quote-aware parsing -- ~2.2% of rows have a quoted comma in 'model' that breaks
naive comma-splitting) then stratified-subsampled via src/subsample_4g5g.py to ~200K rows,
preserving the 4G:5G ratio and full Jan-2024-to-Dec-2025 time range per (tech, month) stratum.
Verified capacity gap: mean download 4G 60,657 kbit vs 5G 230,536 kbit (~3.8x, larger than the
original 3G/4G gap). Reuses the generic pieces from shared_data.py (SingleViewDataset, fit_scaler,
make_load_label, time_blocked_split) -- only the schema-specific loading/labeling/feature-selection
logic is duplicated here.
"""
import os

import numpy as np
import pandas as pd

from shared_data import (  # noqa: F401 -- re-exported for callers
    SingleViewDataset, fit_scaler, make_load_label, time_blocked_split,
)

# Override via env var so the same code works unmodified on Colab (e.g. a Drive-mounted path)
# or anyone else's local checkout without editing this file.
RAW_CSV = os.environ.get("KDCL_RAT_CSV", "data/raw/netztest-4g5g-2024-2025-sample200k.csv")
RATS = ("4G", "5G")

# IDs, device/software metadata, and columns confirmed 100% empty in the original 2016-06 file
# (2026-07-28 EDA) -- not radio-state features. Most of these were never extracted into the new
# 4G/5G file at all (extract_4g5g.py only keeps a minimal column set); harmless to list names that
# no longer exist -- numeric_feature_columns only excludes columns that are actually present.
NON_FEATURE_COLS = {
    "open_uuid", "open_test_uuid", "time_utc", "cat_technology", "network_type", "loc_src",
    "gkz", "zip_code", "country_location", "server_name", "platform", "model", "client_version",
    "network_mcc_mnc", "network_name", "sim_mcc_mnc", "nat_type", "asn", "ip_anonym", "pinned",
    "kg_nr", "gkz_sa", "cell_area_code", "cell_location_id", "channel_number", "radio_band",
    "link_id", "link_name", "link_distance", "edge_id", "link_frc", "dtm_level", "temperature",
    "signal_classification", "atraster100", "atraster250", "implausible", "test_duration",
    "num_threads",
}

# Missingness is tech-conditional and genuinely informative, not a data artifact -- kept as
# features with an explicit missing-flag per column. In the new 4G/5G/2024-2025 slice:
# lte_rsrp/lte_rsrq are ~73.6% populated for 4G, ~54.3% for 5G (5G-NSA phones report their LTE
# anchor carrier's RSRP/RSRQ); signal_strength is now nearly always missing for BOTH techs
# (<0.3%) -- unlike the original 3G/4G slice where it was 3G's primary signal, it's a legacy
# field modern devices rarely report (2026-09-23 EDA). lte_sinr_est (derived in extract_4g5g.py
# from lte_rsrp/lte_rsrq -- an approximation, not a measurement, see that module's docstring) has
# its _missing flag pre-computed during extraction, not here.
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
    """The 4G/5G/2024-2025 file (see module docstring) is already technology-filtered and
    plausible-filtered by extract_4g5g.py, so those two steps are skipped when the columns they'd
    need ('implausible') are already gone. Kept conditionally so load_raw still works standalone
    against a fresh, unfiltered slice in the future."""
    df = pd.read_csv(path, low_memory=False)
    if "cat_technology" in df.columns:
        df = df[df["cat_technology"].isin(rats)].reset_index(drop=True)
    if "implausible" in df.columns:
        df = df[df["implausible"] == False].reset_index(drop=True)  # noqa: E712
    df["is_" + rats[1]] = (df["cat_technology"] == rats[1]).astype(np.float32)
    for col in SPARSE_COLS:
        if col in df.columns:
            df[col + "_missing"] = df[col].isna().astype(np.float32)
    return df


def numeric_feature_columns(df: pd.DataFrame, include_tech_flag: bool = False, rats: tuple = RATS) -> list:
    """Default (include_tech_flag=False) excludes 'is_<rats[1]>' -- it's such a dominant predictor
    (2026-07-28 finding) that both models leaned on it instead of genuinely reasoning from
    radio-condition features, collapsing the RAT-selection recommender to a constant "always
    recommend <rats[1]>" output. include_tech_flag=True keeps it, for the legacy/comparison mode."""
    excluded = NON_FEATURE_COLS | LEAKY_COLS | {LOAD_COL, LABEL_COL}
    if not include_tech_flag:
        excluded = excluded | {"is_" + rats[1]}
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
