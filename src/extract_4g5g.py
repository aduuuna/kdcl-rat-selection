"""One-time extraction: filters the full RTR-NetzTest merged dump (2012-2026, ~12.6GB, every
technology) down to a small 4G/5G, 2024-2025, plausible-only CSV for training.

Uses pandas' C parser (not a naive comma-split) because ~2.2% of rows in the source file have a
quoted comma inside the 'model' field (e.g. device names like "Toshiba ... AT400, Other ... AT300SE"),
which shifts every later column by one position under naive splitting -- confirmed by direct
inspection (2026-09-23 EDA). signal_strength/land_cover/settlement_type/implausible all sit after
'model' in the column order, so this is not a cosmetic distinction: a naive split would have
silently corrupted those columns for ~1.07M rows.

Also derives an APPROXIMATE LTE SINR proxy from lte_rsrp/lte_rsrq for rows where both are present
(mostly 5G-NSA rows anchoring on LTE, plus native 4G rows). This is NOT a measured SINR -- true
SINR requires interference+noise measurements devices don't report. The proxy rearranges the
3GPP TS 36.214 RSRQ definition:
    RSRQ (linear) = N_RB * RSRP (linear) / RSSI (linear)
    => RSSI_dB = 10*log10(N_RB) + RSRP_dB - RSRQ_dB
    => SINR_est_dB = 2*RSRP_dB - RSSI_dB   (treats interference+noise as RSSI-RSRP; RSRP itself is
       the useful-signal term twice over, once directly and once implicitly cancelling RSSI's
       RSRP component -- this is the standard crude approximation used when channel bandwidth is
       unknown, not a physical derivation of true SINR)
    N_RB is unknown here (channel bandwidth isn't in the dataset) -- assumed 50 (10MHz channel,
    the most common LTE channel width), a documented assumption, not a measurement.
"""
import pandas as pd
import numpy as np

SRC = "data/netztest-opendata-ALL-2012-2026.csv"
OUT = "data/raw/netztest-4g5g-2024-2025.csv"

RATS = ("4G", "5G")
YEARS = {"2024", "2025"}

NEEDED_COLS = [
    "time_utc", "cat_technology", "lat", "long", "loc_accuracy", "download_kbit",
    "ping_ms", "lte_rsrp", "lte_rsrq", "implausible", "signal_strength",
    "land_cover", "settlement_type",
]
SPARSE_COLS = ("lte_rsrp", "lte_rsrq", "signal_strength", "land_cover", "settlement_type", "loc_accuracy")
NUMERIC_COLS = ("lat", "long", "loc_accuracy", "download_kbit", "ping_ms", "lte_rsrp", "lte_rsrq")

N_RB_ASSUMED = 50  # 10MHz channel -- documented assumption, see module docstring


def main():
    chunks = []
    total_seen = 0
    reader = pd.read_csv(SRC, usecols=NEEDED_COLS, dtype=str, chunksize=1_000_000)
    for i, chunk in enumerate(reader):
        total_seen += len(chunk)
        mask = (
            chunk["cat_technology"].isin(RATS)
            & (chunk["implausible"] == "false")
            & chunk["time_utc"].str[:4].isin(YEARS)
        )
        survivors = chunk.loc[mask].drop(columns=["implausible"])
        if len(survivors):
            chunks.append(survivors)
        print(f"chunk {i+1}: scanned {total_seen:,} rows so far, kept {sum(len(c) for c in chunks):,}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal rows kept: {len(df):,} (scanned {total_seen:,} total)")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in SPARSE_COLS:
        df[col + "_missing"] = df[col].isna().astype(np.float32)

    df["is_" + RATS[1]] = (df["cat_technology"] == RATS[1]).astype(np.float32)

    has_both = df["lte_rsrp"].notna() & df["lte_rsrq"].notna()
    rssi_db = 10 * np.log10(N_RB_ASSUMED) + df["lte_rsrp"] - df["lte_rsrq"]
    df["lte_sinr_est"] = np.where(has_both, 2 * df["lte_rsrp"] - rssi_db, np.nan)
    df["lte_sinr_est_missing"] = (~has_both).astype(np.float32)

    df = df.sort_values("time_utc").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df):,} rows, {len(df.columns)} columns)")

    print("\n--- class balance (cat_technology) ---")
    print(df["cat_technology"].value_counts())
    print("\n--- lte_rsrp / lte_rsrq / signal_strength completeness by tech ---")
    print(df.groupby("cat_technology")[["lte_rsrp", "lte_rsrq", "signal_strength", "lte_sinr_est"]].apply(
        lambda g: g.notna().mean()
    ))
    print("\n--- download_kbit describe by tech ---")
    print(df.groupby("cat_technology")["download_kbit"].describe())


if __name__ == "__main__":
    main()
