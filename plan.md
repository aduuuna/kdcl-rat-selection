# Plan: KDCL on RTR-NetTest (3G vs 4G) — Fresh Start

Replaces the previous `plan.md`, which was entirely about the original `urban_dataset_processed.xlsx`
research (now superseded — see git history if that phase is ever needed again). See
`docs/notes/progress_log.md` for the full run-by-run log this plan is built from, and
`docs/notes/build_explanation.md` for a code-linked technical walkthrough of how everything works.

## Done

- [x] Found and evaluated candidate datasets, settled on RTR-NetTest Austria (real, verified RAT
      performance gap — see `extradatasets/README.md`).
- [x] EDA on `2016-06` (3G/4G): found and excluded label-leakage columns (`ndt_download_kbit`,
      `ndt_upload_kbit`, `upload_kbit`), confirmed tech-conditional missingness on signal columns,
      settled on `download_kbit` (quantile-binned, 3 classes) as the label.
- [x] Built `src/netztest_data.py` + `src/train_netztest.py`, reusing the generic pipeline pieces
      from `data.py` (`MultiViewDataset`, `fit_scaler`, `time_blocked_split`, `make_load_label`).
- [x] Ran vanilla / DML / KDCL(default) / KDCL(tuned) comparison, 50 epochs each:

  | Mode | 3G_agent | 4G_agent |
  |---|---|---|
  | vanilla | 69.00% | **69.51%** |
  | dml | 67.90% | 68.15% |
  | kdcl (T=1.0, alpha=0.1–0.2, tuned) | 68.89% | 69.45% |

  DML shows real (if modest) negative transfer; KDCL recovers nearly all of it, landing within noise
  of vanilla for the stronger 4G model. The clearest result either dataset has produced.

## Next — the two things asked for now

### 1. Held-out test-set evaluation — DONE
- [x] `src/eval_netztest.py` written: loads saved checkpoints, evaluates on `test_df`, reports
      accuracy per branch per mode alongside the validation numbers.
- [x] Test numbers confirm the validation-based story: DML still clearly worst, tuned KDCL edges
      past vanilla for both branches (3G: 67.66% vs 67.47%; 4G: 68.42% vs 68.37%). See
      `progress_log.md` Run 8 for the full table.

### 2. RAT-selection inference stage — BUILT, found a deeper problem than expected
- [x] `src/infer_netztest.py` written, confound fix applied from day one (forces `is_4G` to each
      branch's own native value before scoring — see `build_explanation.md`).
- [x] Evaluated on the held-out test split with a majority-class baseline. **Result: the
      recommender is degenerate — it recommends "4G" for 100% of test rows.** The 66.1%/99.8%
      headline numbers are mechanical artifacts of that, not real reasoning (see `progress_log.md`
      Run 9 for the full diagnosis).
- [ ] **New follow-up, not yet done:** retrain both branches with `is_4G` *excluded* from the
      feature set, forcing them to learn from the 15 genuine radio-condition features alone, then
      rebuild the recommender on top of those. This is the actual fix, not just a documented
      limitation — worth doing before concluding the recommendation stage doesn't work at all.

## Also outstanding (lower priority, not blocking the two above)

- [ ] Repeated-seed runs — everything so far is a single seed; the tuned-KDCL-vs-vanilla gap
      (0.06 points) is within plausible run-to-run noise and shouldn't be over-claimed without this.
- [ ] Try `KDCL-Linear` (`src/kdcl.py`'s `linear()`) as an alternative to MinLogit — implemented,
      never actually tried on either dataset.
- [ ] ICL ablation (`--distortions none none`) and the noise-robustness test — worth doing properly
      this time since distortions are calibrated in already-normalized space from the start (the old
      dataset had to discover and fix a scale mismatch after the fact; this pipeline doesn't have
      that problem).
- [ ] Per-RAT confusion matrix / F1, not just raw accuracy.
- [ ] Optionally extend to the 2021-06 tri-RAT file (4G/5G/3G) or the 2026-06 4G/5G file for a
      second, independent comparison point — would need a 3-branch extension for the tri-RAT case.

## Write-up
Not started for this dataset. Once the two "next" items above land, worth producing a
`docs/notes/writeup.md` equivalent (thesis-chapter structured), same as was done for the old
dataset before it was superseded.
