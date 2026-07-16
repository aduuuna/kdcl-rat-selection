"""Phase 1 EDA: run with `python notebooks/eda.py` from the repo root.
Checks label leakage, missingness, and class balance before finalizing the KPI label/features.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from data import load_raw, make_load_label, numeric_feature_columns, NON_FEATURE_COLS

pd.set_option("display.width", 120)

df = load_raw()

print("=== NetworkTech distribution ===")
print(df["NetworkTech"].value_counts())
print(df["NetworkTech"].value_counts(normalize=True))

print("\n=== Correlation: derived_load vs candidate leakage sources ===")
candidates = ["temp_load", "sig_load", "DL_bitrate", "UL_bitrate", "SNR", "CQI", "Level"]
candidates = [c for c in candidates if c in df.columns]
print(df[["derived_load"] + candidates].corr()["derived_load"])

print("\n=== derived_load / temp_load / sig_load descriptive stats ===")
print(df[["derived_load", "temp_load", "sig_load"]].describe())

print("\n=== Missingness (% null) across *_missing-flagged source columns ===")
missing_flag_cols = [c for c in df.columns if c.endswith("_missing")]
for flag_col in missing_flag_cols:
    source_col = flag_col.replace("_missing", "")
    if source_col in df.columns:
        print(f"{source_col}: flagged missing {df[flag_col].mean()*100:.1f}% of rows")

print("\n=== Class balance under current quantile-based load_class ===")
df["load_class"] = make_load_label(df)
print(df["load_class"].value_counts(normalize=True))
for rat in ("4G", "5G"):
    sub = df[df["NetworkTech"] == rat]
    print(f"{rat}:", sub["load_class"].value_counts(normalize=True).to_dict())

print("\n=== Current feature columns (after excluding IDs, leaky cols, and the label) ===")
feature_cols = numeric_feature_columns(df)
print(f"count: {len(feature_cols)}")
print(feature_cols)
assert "load_class" not in feature_cols, "label leaked into features!"
assert "temp_load" not in feature_cols and "sig_load" not in feature_cols, "leaky cols not excluded!"
print("OK: label and leaky columns confirmed excluded from features.")
