# Verification of thesis numbers (4G/5G pivot)

Run on 2026-10-22 against this repo, 4G/5G experiments only (`experiments/_archive_3g4g_*` ignored).
Nothing in `src/` was modified. Raw outputs below.

## Summary: thesis claim vs repo

| Item | Thesis says | Repo shows | Verdict |
|---|---|---|---|
| Recommender test size | 29,999 | 29,999 | matches |
| Recommender 4G / 5G | 16,323 / 13,676 | seed 1 only; seeds 2 and 3: 9,161 / 20,838 and 9,367 / 20,632 | single seed |
| Recommender agreement | 58.8% | seed 1 58.8%; seeds 2, 3: 59.9%, 62.0%; mean 60.26 ± 1.32 | single seed |
| Majority baseline | 56.0% | 56.0% (always 5G) | matches |
| Top-load agreement | 62.0% | seed 1 62.0%; seeds 2, 3: 72.7%, 76.1%; mean 70.28 ± 6.02 | single seed |
| Feature count | 17 | 17 | matches |
| Train / val / test | 140,000 / 29,999 / 29,999 | 140,000 / 29,999 / 29,999 | matches |
| 4G Low share | 47.5% | 47.54% | matches |
| 5G High share | 59.6% | 59.57% | matches |
| 4G High share | 13.6% | 13.60% | matches |
| lte_rsrp populated, 4G / 5G | 73.6% / 54.3% | training sample: 73.30% / 54.16%; full 1,411,480-row file: 73.58% / 54.27% | from the full file, not the sample |
| signal_strength populated | under 0.3% | sample: 4G 0.31%, 5G 0.25%, overall 0.28% | 4G is just over 0.3% |
| Total rows, 4G / 5G | 199,998; 114,150 / 85,848 | 199,998; 114,150 / 85,848 | matches |
| Params small / large | 3,523 / 46,851 | 3,523 / 46,851 | matches |
| Architecture | (64,32), (256,128,64), BN, ReLU, Dropout 0.2 | confirmed | matches |
| KDCL T, alpha | 1.0, 0.1, chosen by validation search | 1.0, 0.1 used; no search was run on the 4G/5G data (see step 4) | wording needs changing |
| DML T, alpha | 2.0, 0.5 | 2.0, 0.5 | matches |
| One 50-epoch KDCL run | ~6.4 min | 3m30s to 3m44s | faster than stated |

## 1. Recommender, full test split, per KDCL seed

```
=== 2026-09-23-10-56-30 (seed 1) ===
Test set size: 29999
Recommendation distribution: {'4G': 16323, '5G': 13676}
Agreement with actual serving RAT: 17652/29999 = 58.8% (proxy measure only)
Majority-class baseline (always recommend '5G'): 56.0%
Agreement specifically on rows that actually achieved the top load class: 7046/11366 = 62.0%

=== 2026-09-23-11-00-21 (seed 2) ===
Test set size: 29999
Recommendation distribution: {'4G': 9161, '5G': 20838}
Agreement with actual serving RAT: 17980/29999 = 59.9% (proxy measure only)
Majority-class baseline (always recommend '5G'): 56.0%
Agreement specifically on rows that actually achieved the top load class: 8265/11366 = 72.7%

=== 2026-09-23-11-03-55 (seed 3) ===
Test set size: 29999
Recommendation distribution: {'4G': 9367, '5G': 20632}
Agreement with actual serving RAT: 18604/29999 = 62.0% (proxy measure only)
Majority-class baseline (always recommend '5G'): 56.0%
Agreement specifically on rows that actually achieved the top load class: 8653/11366 = 76.1%
```

Across seeds (mean ± population SD): agreement 60.26 ± 1.32%, top-load agreement 70.28 ± 6.02%,
share recommended 4G 38.72 ± 11.10%. Actual test split: 4G 13,192 (44.0%), 5G 16,807 (56.0%).

**Which seed reproduces the thesis:** seed 1 (`2026-09-23-10-56-30`), exactly. The thesis numbers are
from that one seed, which is the lowest-scoring of the three on both agreement measures.

**Where the numbers came from:** a project-wide search for "16,323", "16323", "58.8", "62.0"
(excluding `.venv/`) finds no file in the working tree. They exist only in git history: commit
`fe5abfb` ("Test results") committed `results/recommender_summary.txt` with the seed-1 full-split
output. The later commit `ed2f161` replaced it with a 500-row run (55.6%, 66.7%) that had been
sitting in the working tree. This file now records all three full-split runs.

## 2. Data facts (`data/raw/netztest-4g5g-2024-2025-sample200k.csv`)

```
--- 2a. numeric_feature_columns ---
 1. lat                 7. signal_strength        13. land_cover_missing
 2. long                8. land_cover             14. settlement_type_missing
 3. loc_accuracy        9. settlement_type        15. loc_accuracy_missing
 4. ping_ms            10. lte_rsrp_missing       16. lte_sinr_est
 5. lte_rsrp           11. lte_rsrq_missing       17. lte_sinr_est_missing
 6. lte_rsrq           12. signal_strength_missing
length: 17

--- 2f. total rows and per-technology counts ---
total rows: 199998
4G    114150
5G     85848

--- 2b. chronological split row counts ---
train=140000  val=29999  test=29999  sum=199998

--- 2c. class balance (0=Low, 1=Medium, 2=High) ---
overall: {0: 33.33, 1: 33.33, 2: 33.33}
4G: {0: 47.54, 1: 38.86, 2: 13.6}
5G: {0: 14.44, 1: 25.98, 2: 59.57}

--- 2d/2e. populated (non-null) share, training sample ---
lte_rsrp 4G: 73.30%
lte_rsrp 5G: 54.16%
lte_rsrp overall: 65.08%
signal_strength 4G: 0.31%
signal_strength 5G: 0.25%
signal_strength overall: 0.28%
```

Same columns in the full extracted file (`data/raw/netztest-4g5g-2024-2025.csv`, 1,411,480 rows;
4G 805,618, 5G 605,862):

```
lte_rsrp 4G: 73.58%
lte_rsrp 5G: 54.27%
lte_rsrp overall: 65.29%
signal_strength 4G: 0.32%
signal_strength 5G: 0.25%
signal_strength overall: 0.29%
```

## 3. Models (17 inputs, 3 classes)

```
rat_mlp_small: trainable params = 3,523
  Linear(17,64) BN ReLU Dropout(0.2) Linear(64,32) BN ReLU Dropout(0.2) Linear(32,3)
  BatchNorm1d layers: 2 | ReLU: 2 | Dropout p: [0.2]

rat_mlp_large: trainable params = 46,851
  Linear(17,256) BN ReLU Dropout(0.2) Linear(256,128) BN ReLU Dropout(0.2)
  Linear(128,64) BN ReLU Dropout(0.2) Linear(64,3)
  BatchNorm1d layers: 3 | ReLU: 3 | Dropout p: [0.2]
```

## 4. Hyperparameters

`args` line from one `train_log.txt` per mode (all three seeds per mode share these values):

```
vanilla     {'mode': 'vanilla', 'ensemble': 'minlogit', 'T': 2.0, 'alpha': 0.5, 'batch_size': 512, 'epochs': 50, 'lr': 0.001, 'weight_decay': 0.0001, 'milestones': [30, 40], 'gamma': 0.1, 'seed': 1, ...}   (T/alpha unused in vanilla)
dml         {'mode': 'dml', 'ensemble': 'minlogit', 'T': 2.0, 'alpha': 0.5, 'batch_size': 512, 'epochs': 50, ...}
kdcl        {'mode': 'kdcl', 'ensemble': 'minlogit', 'T': 1.0, 'alpha': 0.1, 'batch_size': 512, 'epochs': 50, ...}
kdcl_linear {'mode': 'kdcl', 'ensemble': 'linear', 'T': 1.0, 'alpha': 0.1, 'batch_size': 512, 'epochs': 50, ...}
one         {'mode': 'one', 'ensemble': 'minlogit', 'T': 2.0, 'alpha': 0.5, 'gate_hidden': 32, 'gate_lr': 0.001, ...}
offline_kd  {'mode': 'offline_kd', 'T': 2.0, 'alpha': 0.5, 'teacher_ckpt': 'experiments/netztest_vanilla/2026-09-23-10-25-05', ...}
            "Loaded frozen teacher (rat_mlp_large) from experiments/netztest_vanilla/2026-09-23-10-25-05, val_acc=54.438..."
```

**Was T=1.0, alpha=0.1 chosen by a validation search?** Not on the 4G/5G data. The history:

1. **Grid search on the original dataset** (commit `a6c27b9`, 2026-07-16, `src/sweep.py`, since deleted).
   It searched T in {1, 2, 4, 8} × alpha in {0.2, 0.5, 0.8}: 12 combinations, 30 epochs, one seed,
   best validation accuracy, on `urban_dataset_processed.xlsx` (not NetzTest). Winner: T=1.0, alpha=0.2.
   alpha=0.1 was not in that grid.
2. **Manual validation checks on the 3G/4G NetzTest data** (`docs/notes/_archive_3g4g_thesis/progress_log.md`,
   Runs 5 to 7, one seed each). Default T=2.0/alpha=0.5 gave 68.78 / 68.78, T=1.0/alpha=0.2 gave
   68.78 / 69.45, and T=1.0/alpha=0.1 gave 68.89 / 69.45. alpha=0.1 was kept, plateaued.
3. **4G/5G pivot:** T=1.0 and alpha=0.1 were carried over unchanged. No re-tuning was run on the 4G/5G data.

Accurate wording would be: *"T=1.0 and α=0.1 were selected by validation accuracy in earlier
single-seed experiments (a 12-point grid on a previous dataset, then a manual check on an earlier
3G/4G slice of RTR-NetzTest) and carried over to the 4G/5G experiments without re-tuning."* DML,
ONE and offline KD use T=2.0, α=0.5, the train.py defaults.

## 5. Hardware and timing

```
CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz   (2 physical cores, 4 logical)
RAM: 7.9 GB
GPU: Intel(R) HD Graphics 520 (integrated, not CUDA-capable)
OS: Microsoft Windows 11 Pro 10.0.26200
Python: 3.11.9 (64-bit)
PyTorch: 2.13.0+cpu   CUDA available: False   torch threads: 2
numpy 2.4.6 | pandas 3.0.3 | scikit-learn 1.9.0
```

Training was **CPU only**. `train.py` has no device or CUDA handling, and the installed PyTorch
is a CPU build. Library versions are what is installed today. The logs do not record the versions
used on 2026-09-23.

Wall-clock per 50-epoch run (folder-name start time to `train_log.txt` modification time,
including data loading):

```
vanilla      3m50s  3m34s  3m45s
dml          4m06s  3m54s  4m38s
kdcl         3m44s  3m30s  3m30s     <- one KDCL-MinLogit run ~3.5-3.7 min
kdcl_linear  8m34s  9m14s  11m03s
one          4m36s  4m51s  5m06s
offline_kd   1m29s  1m27s  1m34s     (4G student only, frozen teacher)
```

The full recommender pass over 29,999 test rows takes 39 to 49 s per seed.

## 6. Figures

`docs/figures/fig_test_accuracy.pdf`, `fig_robustness_4g.pdf`, `fig_recommender_distribution.pdf`,
regenerated from the current `results/summary.csv`, `results/robustness.csv` and the three
recommender runs above, with a white background. A text search of all three PDFs finds no "3G".
The accuracy bars start at zero. The recommender figure shows all three seeds next to the actual
serving split, not seed 1 alone.
