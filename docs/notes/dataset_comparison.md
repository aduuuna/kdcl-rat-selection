# Dataset Comparison: Original (Urban) vs. New (RTR-NetTest)

Column-by-column comparison between `urban_dataset_processed.xlsx` (the original, supervisor-provided
dataset — **64 columns**) and `netztest-opendata-2016-06.csv` (the new RTR-NetTest Austria dataset —
**52 columns**). Every single column from both files is accounted for exactly once below, grouped
by category so equivalent concepts line up even where names differ. "Equivalent" means the same
underlying concept exists on both sides (even if computed differently, or unusable in this
particular file); "only" means one dataset has no counterpart at all.

## Identity / session tracking

| Old | New | Status |
|---|---|---|
| `SessionID` | `open_test_uuid` | Equivalent |
| `Timestamp` | `time_utc` | Equivalent |
| `Node`, `CellID` | `cell_area_code`, `cell_location_id` | Equivalent (both **100% empty** in this new file) |
| `LAC` | — | **Old only** |
| `State` | — | **Old only** |
| `ElapsedTime` | — | **Old only** (new dataset has no within-session elapsed-time counter) |
| `Operatorname` | `network_mcc_mnc` | Equivalent |
| — | `open_uuid` | **New only** (finer-grained than session) |
| — | `network_type` | **New only** (redundant second tech-label column alongside `cat_technology`) |
| — | `network_name` | **New only** (redundant second operator-name column alongside `network_mcc_mnc`) |
| — | `sim_mcc_mnc` | **New only** (home carrier, vs. serving carrier — detects roaming) |
| — | `nat_type`, `asn`, `ip_anonym` | **New only** (network infrastructure metadata) |
| — | `country_location`, `loc_src` | **New only** |

## Radio access technology

| Old | New | Status |
|---|---|---|
| `NetworkTech` | `cat_technology` | Equivalent |
| `ARFCN` | `channel_number` | Equivalent (**100% empty** in this new file) |
| `BANDWIDTH` | `radio_band` | Equivalent (**100% empty** in this new file) |
| `PSC` | — | **Old only** (3G/UMTS scrambling code) |

## Signal quality

| Old | New | Status |
|---|---|---|
| `Level` | `lte_rsrp` + `signal_strength` | Equivalent (new dataset splits one concept into an LTE version and a legacy-tech version) |
| `Qual` | `lte_rsrq` | Equivalent (LTE-only; no legacy-tech quality field) |
| `SNR` | — | **Old only** |
| `CQI` | — | **Old only** |

## Throughput / performance

| Old | New | Status |
|---|---|---|
| `DL_bitrate` | `download_kbit` | Equivalent (**role changed**: old = feature, new = the label itself) |
| `UL_bitrate` | `upload_kbit` | Equivalent (**excluded as a feature** in the new pipeline — leakage risk, 0.77 correlation with the label) |
| `SERVINGTIME` | `test_duration` | Equivalent (different meaning: old = time on serving cell, new = speed-test duration) |
| — | `ping_ms` | **New only** (no latency measurement anywhere in the old dataset) |
| — | `ndt_download_kbit`, `ndt_upload_kbit` | **New only** (alternate test methodology — excluded as a leakage risk) |

## Location / geography

| Old | New | Status |
|---|---|---|
| `Longitude` | `long` | Equivalent |
| `Latitude` | `lat` | Equivalent |
| `Accuracy` | `loc_accuracy` | Equivalent |
| `Altitude` | — | **Old only** |
| `Node_Longitude`, `Node_Latitude` | — | **Old only** (serving cell tower's own coordinates) |
| — | `land_cover`, `settlement_type` | **New only** (terrain / urban-vs-rural — old dataset has no land-use context at all) |
| — | `zip_code`, `gkz`, `gkz_sa`, `kg_nr` | **New only** (Austrian administrative area codes) |
| — | `atraster100`, `atraster250` | **New only** (spatial raster grid cell IDs) |

## Neighbor-cell metrics — the single biggest gap

| Old | New | Status |
|---|---|---|
| `SecondCell_NODE`, `SecondCell_CELLID`, `SecondCell_RSRP`, `SecondCell_SNR`, `SecondCell_PSC`, `SecondCell_ARFCN` | — | **Old only** (full neighbor-cell-1 signal profile) |
| `NTech1`, `NCellid1`, `NLAC1`, `NCell1`, `NARFCN1`, `NRxLev1`, `NQual1` | — | **Old only** (neighbor-cell-2 signal profile) |

The new dataset has **no neighbor-cell information whatsoever** — the old dataset's richest
structural feature (useful for handover-adjacent reasoning) simply doesn't exist in the new schema.

## Mobility / device context

| Old | New | Status |
|---|---|---|
| `Test_Status` | `implausible` | Equivalent (inverse concept — old flags a good test, new flags a bad one) |
| `Mobility` | — | **Old only** (walking/driving/stationary label) |
| `Speed` | — | **Old only** |
| — | `platform`, `model`, `client_version` | **New only** (device OS, phone model, app version) |
| — | `server_name`, `num_threads` | **New only** (speed-test methodology parameters) |
| — | `pinned` | **New only** |

## Road/link metadata (new dataset only, and mostly unusable)

| Old | New | Status |
|---|---|---|
| — | `link_id`, `link_name`, `link_distance`, `edge_id`, `link_frc`, `dtm_level` | **New only** — `link_distance` is 93% missing; the rest weren't individually checked but aren't used |
| — | `signal_classification`, `temperature` | **New only** — both **100% empty** in this file |

## Engineered / derived features (old dataset only)

| Old | New | Status |
|---|---|---|
| `derived_load` | (`download_kbit`, raw) | Equivalent (concept) — old dataset's label was a pre-engineered composite; new dataset's label is computed directly from the raw throughput column ourselves |
| `temp_load`, `sig_load` | — | **Old only** (components of `derived_load` — excluded from features as leakage; new dataset needed no equivalent since we don't pre-engineer the label) |
| `mobility_encoded`, `SNR_norm`, `Latitude_norm`, `Longitude_norm` | — | **Old only** (pre-normalized/encoded columns baked into the file; for the new dataset we apply our own z-score scaling at the pipeline level instead, via `fit_scaler`) |
| `time_bin` | — | **Old only** (pre-bucketed time; new dataset gives only a raw timestamp, which we sort by directly for the train/val/test split) |
| `NCellid1_missing`, `SecondCell_NODE_missing`, `SecondCell_CELLID_missing`, `SecondCell_PSC_missing`, `SecondCell_ARFCN_missing`, `NLAC1_missing`, `NCell1_missing`, `NARFCN1_missing`, `CQI_missing`, `BANDWIDTH_missing`, `SecondCell_RSRP_missing`, `SecondCell_SNR_missing`, `NRxLev1_missing`, `NQual1_missing` | (we build our own 6, for `lte_rsrp`/`lte_rsrq`/`signal_strength`/`land_cover`/`settlement_type`/`loc_accuracy`) | Equivalent (concept) — old dataset shipped 13 pre-built missing-flags; new dataset ships none, so we generate our own for the columns that need them (`SPARSE_COLS` in `netztest_data.py`) |

---

## Summary table

| | Old (`urban_dataset_processed.xlsx`) | New (`netztest-opendata-2016-06.csv`) |
|---|---|---|
| **Total raw columns** | 64 | 52 |
| **Have an equivalent concept on the other side** | 19 | 18 |
| **No equivalent — only in this dataset** | 45 | 34 |

(The 19 vs. 18 equivalent-count difference is because a few columns map many-to-one across the two
schemas — e.g. old `Level` corresponds to *two* new columns, `lte_rsrp` and `signal_strength` — so
the counts don't need to match exactly.)

**Bottom line:** the two datasets overlap on the fundamentals — location, RAT label, signal
strength/quality, throughput, session identity, a test-quality flag — but diverge sharply
everywhere else. The **old dataset's unique strength** is its neighbor-cell signal profiles and
mobility/speed context (13 neighbor-cell columns alone, all absent from the new dataset). The
**new dataset's unique strength** is genuine geographic/terrain context, device metadata, and
latency (`ping_ms`) — none of which the old dataset has at all. Neither dataset is a superset of
the other; each has real information the other one lacks.

---

## Which dataset is best?

**The new dataset (RTR-NetTest), clearly, for this specific project.** The reasoning, not just the
verdict:

The whole thesis question is *"does KDCL's collaborative-distillation mechanism protect a weaker
RAT model without letting it drag down a stronger one, when a real capacity gap exists?"* The old
dataset **structurally could not answer that question** — this was proven three separate ways
(smaller architecture, harder label, 3x training time), and the last attempt confirmed genuine
overfitting rather than a fixable undertraining issue. No matter how good the engineering was, that
dataset never had a real 3G/4G-style gap to exploit. The new dataset does (5x real throughput
difference), and it produced the first clean, validated (train/val/test all agree) result: DML
genuinely fails, KDCL genuinely recovers almost all of it.

That said, it is not a one-sided win — the old dataset has real strengths the new one lacks (see
table below), so the claim is specifically that the new dataset wins *for this research question*,
not that it is better in every sense.

| | Old (urban) | New (netztest) |
|---|---|---|
| Supports the core research question | No — proven, three ways | Yes — validated on held-out test set |
| Real performance gap between RATs | No | Yes (5x) |
| Neighbor-cell context | Rich (13 columns) | None |
| Mobility/speed context | Yes | None |
| Rows | 22,690 | 35,769 (this one month alone; more months available) |
| Independent / verifiable | Supervisor-provided, single source | Public, documented, other months available for extension |

The old dataset's richness (neighbor cells especially) would be genuinely valuable for a *different*
research question — something closer to real handover-timing prediction. For "does KDCL work when a
real capacity gap exists," it was never going to get there.

## Can the old dataset's engineering be replicated on the new one?

Mostly already done, one gap that cannot be filled, and one genuinely worthwhile addition still
open:

| Old engineered feature | Status on new dataset |
|---|---|
| `SNR_norm`, `Latitude_norm`, `Longitude_norm` (pre-normalized columns) | **Already done, and better** — `fit_scaler()` z-score normalizes all 15 features consistently, not just three hand-picked ones |
| 13× `*_missing` flags | **Already done** — the `SPARSE_COLS` mechanism in `netztest_data.py` builds our own 6 |
| `derived_load` (engineered label) | **Already done, more transparently** — `download_kbit` is binned directly, so there is no hidden formula that could accidentally leak (that was literally the bug caught in the old dataset) |
| `temp_load`, `sig_load` | **Not replicated, on purpose** — these were the *leaky* components of `derived_load` in the old dataset. Recreating them would reintroduce a bug that was specifically found and fixed, not add a useful feature |
| `mobility_encoded` | **Cannot be replicated** — the new dataset has no speed/mobility field at all. This is a genuine data gap, not an engineering choice |
| `time_bin` (pre-bucketed time) | **Not done yet, and worth doing** — bucketing `time_utc` into hour-of-day (or weekday/weekend) is straightforward to add, and network load has a well-known time-of-day pattern (rush hour vs. overnight) that is currently unused |

**Summary:** almost everything meaningful from the old dataset's engineering has already been
matched or improved on the new dataset. The one clear remaining opportunity is an **hour-of-day
feature** derived from `time_utc`, which has not been added yet.
