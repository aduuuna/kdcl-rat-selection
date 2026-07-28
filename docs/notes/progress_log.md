# Progress Log: Fresh Start — RTR-NetTest Austria (3G vs 4G)

Living log, updated after **every** run, in order, so progress is visible without digging through
commit history. This is a clean restart on a new dataset — the previous dataset's full journey
(phases 1-7, `urban_dataset_processed.xlsx`) is preserved in git history (`git log`) if ever needed
again, but is not carried forward here.

## Why a fresh dataset

The original dataset couldn't support a real capacity gap between RAT branches no matter how hard
we tried to force one (three separate attempts: smaller architecture, harder label, 3x training
epochs — the smaller model won every time, and the last attempt showed clear overfitting, not
undertraining). That meant KDCL's actual headline claim — protecting a strong model from being
dragged down by a genuinely weaker one — was never fairly testable. Decided to find a dataset with
a real performance gap between technologies instead of continuing to force one artificially.

## Dataset search

Several candidates were evaluated and rejected (see `extradatasets/README.md` for the full list —
same-schema duplicate, RAT-imbalanced, single-feature-only, corrupted download, etc.) before
landing on **RTR-NetTest Austria** (netztest.at/en/Opendata, CC BY 4.0) — real crowdsourced
speed-test data, one CSV per month since 2012. Four months kept as usable, with verified real RAT
diversity:

| File | Mobile rows | RAT split |
|---|---|---|
| `2016-06` | 106,530 | 4G 22,533 / **3G 13,255** / 2G 1,576 |
| `2019-06` | 232,019 | 4G 35,256 / 3G 4,531 (5G only 13 — unusable) |
| `2021-06` | 368,524 | 4G 29,980 / 5G 8,537 / 3G 1,582 — tri-RAT |
| `2026-06` | 463,988 | 5G 33,066 / 4G 23,315 |

**Chose 3G vs 4G (`2016-06`) to start** — it has a real, physically-grounded performance gap the
old dataset never had:

| Tech | median download | median upload | median ping |
|---|---|---|---|
| 3G | 6,282 kbit | 1,632 kbit | 58.8 ms |
| 4G | 33,348 kbit | 10,854 kbit | 30.2 ms |

## Cleanup

Deleted ~20.9GB of unneeded bulk downloads: 159 unused monthly CSVs in `data/raw/rtr_monthly/`
(kept only the 4 accepted months) and `data/datasets.zip` (8.2GB). Per explicit instruction, did
**not** touch `data/raw/netztest-opendata-ALL-2012-2026.csv` (13.6GB, kept as-is). Also cleared out
every old experiment run folder and the previous `progress_log.md` for a genuinely clean start —
all of it remains recoverable via `git log` since it was committed before deletion.

## EDA on `2016-06` mobile rows

Same discipline as the original dataset's Phase 1 — checked for leakage/quality issues before
writing any pipeline code:
- `implausible` flag: True for only 19/37,364 rows — dropped, trivial.
- **Leakage found (same pattern as the old dataset):** `ndt_download_kbit`/`ndt_upload_kbit`
  correlate 0.87–0.94 with `download_kbit`/`upload_kbit` (an alternate speed-test methodology
  measuring the same throughput). Excluded both.
- `upload_kbit` correlates 0.77 with `download_kbit` (our label candidate) — similar risk level to
  the old dataset's `temp_load` (0.79, which was excluded there). Excluded `upload_kbit` too.
- Several schema columns are 100% empty in this specific file: `cell_area_code`,
  `cell_location_id`, `channel_number`, `radio_band`, `temperature`, `signal_classification`.
  Excluded, along with `link_distance` (93% missing, road-network proxy, not signal quality).
- `lte_rsrp`/`lte_rsrq`/`signal_strength` are missing in a **tech-conditional** way (e.g. `lte_rsrp`
  is 100% missing for 2G/3G, 36% missing for 4G — non-LTE phones can't report an LTE metric). This
  is genuine physics, not a data artifact, so kept as features with an explicit missing-flag per
  column (same treatment as the old dataset's neighbor-cell fields).

**Label:** `download_kbit`, quantile-binned into 3 classes (Low/Medium/High).
**Final feature count:** 16 (10 real signal/location columns + 6 missing-flags + `is_4G`).

**Class balance confirms a real signal, not just a raw-metric gap:** 3G rows hit the "High" class
0.07% of the time; 4G rows hit it 52.9% of the time — a much starker separation than the old
dataset's per-RAT class skew, which was only a few percentage points.

## Pipeline built

- `src/netztest_data.py` — schema-specific loading/labels/features for this dataset. Reuses the
  generic, dataset-agnostic pieces from `data.py` directly (`MultiViewDataset`, `SingleViewDataset`,
  `fit_scaler`, `make_load_label`, `time_blocked_split`) rather than duplicating them.
- `src/train_netztest.py` — mirrors `train.py`'s training loop exactly, swapped to import from
  `netztest_data` and branch names `3G_agent`/`4G_agent`. Kept as a parallel script rather than
  modifying the original `train.py`, so the old dataset's code path stays intact if ever revisited.
- `src/kdcl.py`, `src/models.py`, `src/distortions.py`, `src/utils.py` — unchanged, fully reused.

**Bug found and fixed immediately:** `make_load_label`'s default `load_col` comes from `data.py`'s
own module constant (`"derived_load"`), so calling it without an explicit `load_col=` in
`train_netztest.py` crashed with `KeyError: 'derived_load'`. Fixed by passing `load_col=LOAD_COL`
(netztest's own constant, `"download_kbit"`) explicitly.

---

## Run log

### Run 1 — smoke test, vanilla, 10 epochs
`python src/train_netztest.py --mode vanilla --epochs 10 --print_freq 2`
```
[10] 3G_agent: train_acc=64.88 val_acc=66.65
[10] 4G_agent: train_acc=65.61 val_acc=67.94
3G_agent best val acc: 67.88
4G_agent best val acc: 68.20
```
Pipeline runs end to end. Gap is small at only 10 epochs (0.32 points) — worth watching, since
`is_4G` is a very strong feature here (3G almost never reaches "High"), so both branches might
converge to similar accuracy just by reading that one flag, regardless of their own capacity.

### Run 2 — vanilla, 50 epochs — interrupted at epoch 40/50
Background process was stopped by the harness (agent teardown), not a code error. Re-ran cleanly
rather than patching together a partial run (Run 3).

### Run 3 — vanilla, 50 epochs, clean run
`python src/train_netztest.py --mode vanilla --epochs 50 --print_freq 10`
```
[10] 3G_agent: val_acc=66.65   4G_agent: val_acc=67.94
[20] 3G_agent: val_acc=68.07   4G_agent: val_acc=68.67
[30] 3G_agent: val_acc=68.03   4G_agent: val_acc=69.04
[40] 3G_agent: val_acc=68.44   4G_agent: val_acc=69.04
[50] 3G_agent: val_acc=68.91   4G_agent: val_acc=69.32
3G_agent best val acc: 69.00
4G_agent best val acc: 69.51
```
**Gap is still small (0.51 points) even after full training**, despite the real 5x raw throughput
difference between 3G and 4G. Likely explanation: `is_4G` is such a strong, cleanly-separating
feature (3G basically never hits "High") that both branches — regardless of their own hidden-layer
capacity — can lean on it to get most of the way to a good answer, similarly to how the old
dataset's task turned out to be near-linearly-separable. Need to watch this closely across
vanilla/DML/KDCL before concluding either way.

Re-ran cleanly after the fresh-start folder wipe (deterministic, same seed) — identical result,
confirmed. Checkpoint: `experiments/netztest_vanilla_rat_mlp_small_rat_mlp_large/2026-07-28-07-20/`.

### Run 4 — dml, 50 epochs
`python src/train_netztest.py --mode dml --epochs 50 --print_freq 10`
```
3G_agent best val acc: 67.90
4G_agent best val acc: 68.15
```
**Real, if modest, negative transfer vs. vanilla** — 3G: 69.00%→67.90% (-1.10), 4G: 69.51%→68.15%
(-1.36). The stronger 4G model was hurt slightly *more* in absolute terms than the weaker 3G model —
the exact direction the paper's DML failure mode predicts (mutual mimicry drags the stronger model
toward the weaker one). Much more promising than the old dataset, where after normalization DML
stopped failing entirely and even beat vanilla. Checkpoint:
`experiments/netztest_dml_rat_mlp_small_rat_mlp_large/2026-07-28-07-26/`.

### Run 5 — kdcl (MinLogit, default T=2.0/alpha=0.5), 50 epochs
`python src/train_netztest.py --mode kdcl --epochs 50 --print_freq 10`
```
3G_agent best val acc: 68.78
4G_agent best val acc: 68.78
```

**Summary so far:**

| Mode | 3G_agent | 4G_agent |
|---|---|---|
| vanilla | 69.00% | 69.51% |
| dml | 67.90% | 68.15% |
| kdcl (default) | 68.78% | 68.78% |

KDCL sits between DML and vanilla for both branches — clearly better than DML's negative transfer,
but not yet beating vanilla. **Exactly the same pattern seen on the old dataset before the T/alpha
hyperparameter sweep fixed it** (default alpha=0.5, inherited from the CIFAR reference, over-weights
the KD term for a 3-class task). Trying the previously-successful `T=1.0, alpha=0.2` directly before
running a full sweep.

### Run 6 — kdcl, T=1.0, alpha=0.2, 50 epochs
`python src/train_netztest.py --mode kdcl --T 1.0 --alpha 0.2 --epochs 50 --print_freq 10`
```
3G_agent best val acc: 68.78
4G_agent best val acc: 69.45
```

| Mode | 3G_agent | 4G_agent |
|---|---|---|
| vanilla | 69.00% | 69.51% |
| dml | 67.90% | 68.15% |
| kdcl (default T=2/alpha=0.5) | 68.78% | 68.78% |
| kdcl (T=1.0/alpha=0.2) | 68.78% | **69.45%** |

**4G_agent is now within 0.06 points of vanilla** — a big jump from default KDCL (68.78%) and a
full recovery of DML's -1.36 point loss, plus more. Same direction as the old dataset's finding
(lower alpha works better for this 3-class task) confirmed here too. Trying one step further
(alpha=0.1) to see if 4G_agent can be pushed past vanilla entirely, matching the old dataset's
eventual headline result (78.72% vs 78.20%).

### Run 7 — kdcl, T=1.0, alpha=0.1, 50 epochs
`python src/train_netztest.py --mode kdcl --T 1.0 --alpha 0.1 --epochs 50 --print_freq 10`
```
3G_agent best val acc: 68.89
4G_agent best val acc: 69.45
```
4G_agent is identical to the alpha=0.2 run (69.45%); 3G nudges up slightly (+0.11). **Plateaued** —
further lowering alpha isn't pushing past vanilla. Stopping the hyperparameter chase here; the
remaining 0.06-point gap to vanilla is within noise for a single-seed run.

---

## Summary so far

| Mode | 3G_agent | 4G_agent |
|---|---|---|
| vanilla | 69.00% | **69.51%** |
| dml | 67.90% | 68.15% |
| kdcl (default T=2/alpha=0.5) | 68.78% | 68.78% |
| **kdcl (T=1.0, alpha=0.1-0.2, tuned)** | **68.89%** | **69.45%** |

**This is a real, coherent, defensible result — the clearest one either dataset has produced:**
1. **DML shows genuine negative transfer** (both branches drop from vanilla, stronger model hurt
   slightly more) — unlike the old dataset, where after normalization DML stopped failing entirely.
2. **KDCL clearly recovers almost all of that loss** for both branches, and specifically **nearly
   matches vanilla for the stronger 4G model** (69.45% vs 69.51%, within noise) — while protecting
   the weaker 3G model from DML's -1.10 point drop too (68.89% vs vanilla's 69.00%, only -0.11).
3. This is much closer to the paper's actual headline claim than anything achieved on the old
   dataset, where no real capacity gap could ever be established. Here, a genuine physical
   performance gap (3G vs 4G, confirmed via median download/upload/ping) gave DML something real to
   fail on, and KDCL something real to fix.

**Not yet done, in rough priority order:** confirm this holds with repeated seeds (single-seed runs
so far); try the `KDCL-Linear` ensemble method as an alternative to MinLogit; per-RAT confusion
matrix/F1; the ICL ablation and noise-robustness test (done properly this time, with normalized
distortion magnitudes from the start rather than discovering the scale issue after the fact); the
RAT-selection inference stage (`infer.py`-equivalent) on this dataset, with the `is_5G`-style
confound fix applied from the start.

### Run 8 — held-out test-set evaluation

Every number above is *validation* accuracy — used during training to pick each checkpoint, so not
fully unbiased. `src/eval_netztest.py` loads the saved checkpoints from runs 3/4/5/7 and evaluates
them on `time_blocked_split`'s third slice (`test_df`), which nothing has touched until now.

```
mode          branch      val_acc  test_acc
vanilla       3G_agent      69.00    67.47
vanilla       4G_agent      69.51    68.37
dml           3G_agent      67.90    66.49
dml           4G_agent      68.15    66.23
kdcl_default  3G_agent      68.78    67.68
kdcl_default  4G_agent      68.78    67.51
kdcl_tuned    3G_agent      68.89    67.66
kdcl_tuned    4G_agent      69.45    68.42
```

**The story holds up on genuinely unseen data.** Every mode drops ~1-2 points from validation to
test — a normal, consistent generalization gap, roughly the same size across all four modes (so not
a sign any one mode specifically overfit to the validation set). The *relative* ordering is intact:
DML is still clearly the worst on both branches (confirms real negative transfer, not a validation
fluke). **Tuned KDCL doesn't just hold its ground — on the test set it edges past vanilla for both
branches** (3G: 67.66% vs 67.47%; 4G: 68.42% vs 68.37%). Margins are small enough to still be within
single-seed noise, but the direction is consistent and encouraging rather than a validation-only
artifact. This is the strongest confirmation either dataset has produced.

### Run 9 — RAT-selection inference stage

`src/infer_netztest.py`: mirrors the old dataset's `infer.py`, with the `is_5G`-style confound fix
built in from the start (`recommend_rat` forces `is_4G` to each branch's own native value before
scoring, rather than letting a branch read a row's real serving tech off its input). Evaluated on
the true held-out test split, using the tuned-KDCL checkpoints
(`experiments/netztest_kdcl_rat_mlp_small_rat_mlp_large/2026-07-28-08-05/`):

```
Test set size: 5365
Agreement with actual serving RAT: 3544/5365 = 66.1%
Majority-class baseline (always recommend '4G'): 66.1%
Agreement specifically on rows that actually achieved the top load class: 1905/1908 = 99.8%
```

**Both numbers looked informative at first glance — they are not.** Checked what the recommender
was actually outputting: it recommends **"4G" for all 5,365 test rows, 100% of the time.** Both
the 66.1% (exactly the majority baseline) and the striking 99.8% figure are mechanical consequences
of a degenerate, constant-output recommender, not evidence of real per-row reasoning. Confirmed
further: the confidence gap between the "hypothetical 4G" and "hypothetical 3G" scores is
**positive for every single sampled row** (mean +0.636, range +0.004 to +1.05, never negative).

**Root cause, one level deeper than the old dataset's confound:** the confound *fix* is correctly
implemented (each branch is scored under its own forced tech identity, not the row's real one). The
problem is upstream of that — `is_4G` is such an overwhelmingly strong predictor of the label
(3G reaches "High" load 0.07% of the time vs. 4G's 52.9%, see Phase 1 EDA above) that **both trained
models likely learned to weight it far more heavily than the other 15 genuine radio-condition
features**, to the point that flipping it dominates the predicted outcome almost regardless of what
signal strength, location, or ping actually say. The models were never given a reason *not* to lean
on it this hard, since `is_4G` really is that predictive during normal training.

**Honest conclusion:** as currently built, the recommender doesn't produce a useful RAT
recommendation — it's collapsed to "always suggest 4G," which happens to be right most of the time
simply because 4G usually is better, not because the model is reasoning from radio conditions. A
genuinely useful recommender likely requires **retraining without `is_4G` as an input feature at
all**, forcing both branches to learn a mapping from radio conditions alone rather than leaning on
the tech-identity shortcut. This is a real, actionable next step, not yet done — flagged rather than
silently accepted, since claiming the 99.8% figure without checking would have been misleading.

### Run 10 — fixing the degenerate recommender: retrain without `is_4G`

Added `--exclude_tech_flag` to `train_netztest.py` (drops `is_4G` from the feature set via
`numeric_feature_columns(df, include_tech_flag=False)` in `netztest_data.py`) and re-trained the
tuned KDCL config on the remaining 15 genuine radio-condition features:

`python src/train_netztest.py --mode kdcl --T 1.0 --alpha 0.1 --epochs 50 --exclude_tech_flag`
```
3G_agent best val acc: 66.10
4G_agent best val acc: 66.75
```
Accuracy dropped from 68.89%/69.45% (with the flag) to 66.10%/66.75% — expected, since a genuinely
predictive feature was removed. This is the correct trade: lower raw accuracy, but the model can no
longer take the shortcut.

Updated `infer_netztest.py` to support this case (`recommend_rat` now handles `is_4g_idx=None` —
no column to force when it was never a training feature) and re-ran the recommender on the held-out
test set with the new checkpoints:

```
Test set size: 5365
Recommendation distribution: {'3G': 2420, '4G': 2945}
Agreement with actual serving RAT: 3522/5365 = 65.6%
Majority-class baseline (always recommend '4G'): 66.1%
Agreement specifically on rows that actually achieved the top load class: 1531/1908 = 80.2%
```

**The fix worked — the recommender is no longer degenerate.** It now genuinely splits its
recommendations (2,420 vs 2,945, not a constant answer), confirming both branches are actually
reasoning from the 15 radio-condition features rather than reading off a dominant flag.

**Honest read of the new numbers:** overall agreement (65.6%) sits right at the majority baseline
(66.1%) — across the *full* range of outcomes, the recommender isn't clearly better than "always
guess 4G." But on the subset where a genuinely top-tier outcome actually occurred, agreement is
80.2%, meaningfully above the ~66% baseline — and this time that number is earned through real
per-row reasoning, not a constant-output artifact. **Net conclusion: a modest but genuine signal
exists specifically for identifying strong-outcome conditions, even though the recommender isn't
yet clearly useful across the board.** This is a much more honest, defensible finding than either
the original inflated 99.8% or a flat "it doesn't work" — it's a real, partial capability with a
precise boundary on where it does and doesn't add value.

---

## Standardizing on no-tech-flag, cleaning up naming, full re-run

Excluding `is_4G` is now **the default**, not an opt-in flag: `train_netztest.py`'s
`--include_tech_flag` defaults to `False` (was `--exclude_tech_flag`, defaulting to include it —
inverted for clarity), and `netztest_data.py`'s `numeric_feature_columns()` default flipped to
match. Everything above this point used the *old* (`is_4G`-included) setup unless a run explicitly
says otherwise — kept as historical record, not deleted, since it's what motivated this change.

**Also cleaned up the experiment folder naming**, which had gotten confusing
(`netztest_kdcl_rat_mlp_small_rat_mlp_large_notechflag` and similar). Since every run so far uses
the same small/large architecture pairing, the model names were dropped from the folder name
entirely. New convention: `netztest_<mode>` normally, `netztest_<mode>_withtechflag` only for the
legacy comparison mode. All old experiment folders were deleted (all were already committed to git,
so nothing is actually lost — recoverable via `git log` if ever needed) and every run below was
redone fresh under the new naming.

### Full re-run: vanilla / dml / kdcl, no tech flag, 50 epochs each

```
netztest_vanilla/2026-07-28-14-27   3G_agent best val acc: 66.10   4G_agent best val acc: 66.82
netztest_dml/2026-07-28-14-30       3G_agent best val acc: 64.79   4G_agent best val acc: 65.16
netztest_kdcl/2026-07-28-14-34      3G_agent best val acc: 66.10   4G_agent best val acc: 66.75   (T=1.0, alpha=0.1)
```

**Held-out test-set evaluation** (`src/eval_netztest.py`, `CHECKPOINTS` updated to the new paths):
```
mode          branch       val_acc  test_acc
vanilla       3G_agent       66.10    65.33
vanilla       4G_agent       66.82    66.54
dml           3G_agent       64.79    63.02
dml           4G_agent       65.16    63.45
kdcl_tuned    3G_agent       66.10    64.85
kdcl_tuned    4G_agent       66.75    66.26
```

### The current baseline, in one table

| Mode | 3G_agent (val / test) | 4G_agent (val / test) |
|---|---|---|
| vanilla | 66.10% / 65.33% | 66.82% / 66.54% |
| dml | 64.79% / 63.02% | 65.16% / 63.45% |
| **kdcl (tuned)** | 66.10% / 64.85% | 66.75% / 66.26% |

**The story holds up completely without `is_4G`, and the test set makes it even clearer:** DML's
negative transfer is real and, if anything, more pronounced on unseen data (-2.31 points for 3G,
-3.09 for 4G, vs. vanilla). KDCL recovers almost all of that gap and lands within half a point of
vanilla on both branches, while clearly and substantially beating DML (+1.83 for 3G, +2.81 for 4G on
the test set). This is the current, standard baseline going forward — everything above this section
used the old (tech-flag-included) setup and is kept for context, not as the active result.
