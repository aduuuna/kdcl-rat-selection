"""Stratified subsample of the full 4G/5G 2024-2025 extraction (1,411,480 rows) down to a
compute-tractable size for iteration and the multi-seed/multi-method sweep. Stratifies by
(cat_technology, year-month) so the original 4G:5G ratio (~57:43) and temporal distribution
are both preserved -- important because train.py's time_blocked_split relies on every
period being represented, not just the overall row count.
"""
import pandas as pd

SRC = "data/raw/netztest-4g5g-2024-2025.csv"
OUT = "data/raw/netztest-4g5g-2024-2025-sample200k.csv"
TARGET_ROWS = 200_000
SEED = 42


def main():
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df):,} rows from {SRC}")

    df["year_month"] = df["time_utc"].str[:7]
    frac = TARGET_ROWS / len(df)
    sample = df.groupby(["cat_technology", "year_month"], group_keys=False)[df.columns].apply(
        lambda g: g.sample(frac=frac, random_state=SEED)
    )
    sample = sample.drop(columns=["year_month"]).sort_values("time_utc").reset_index(drop=True)

    print(f"Sampled {len(sample):,} rows (target was {TARGET_ROWS:,}, frac={frac:.4f})")
    sample.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")

    print("\n--- class balance ---")
    print(sample["cat_technology"].value_counts())
    print("\n--- class balance, full population (for comparison) ---")
    print(df["cat_technology"].value_counts())
    print("\n--- time range check ---")
    print("full:", df["time_utc"].min(), "to", df["time_utc"].max())
    print("sample:", sample["time_utc"].min(), "to", sample["time_utc"].max())


if __name__ == "__main__":
    main()
