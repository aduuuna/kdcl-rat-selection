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
- [x] Ran vanilla / DML / KDCL(default) / KDCL(tuned) comparison, 50 epochs each, **with `is_4G`
      included as a feature** — see below, this setup has since been superseded.
- [x] **Standardized on excluding `is_4G` from the feature set** (see item 2 below for why —
      it made the RAT-selection recommender degenerate). This is now the default in
      `train_netztest.py` (`--include_tech_flag` opts back into the old behavior). Full comparison
      re-run under this standard:

  | Mode | 3G_agent (val / test) | 4G_agent (val / test) |
  |---|---|---|
  | vanilla | 66.10% / 65.33% | 66.82% / 66.54% |
  | dml | 64.79% / 63.02% | 65.16% / 63.45% |
  | **kdcl (T=1.0, alpha=0.1, tuned)** | 66.10% / 64.85% | 66.75% / 66.26% |

  DML shows real negative transfer, more pronounced on the held-out test set than on validation
  (-2.31/-3.09 points vs. vanilla). KDCL recovers almost all of it and lands within half a point of
  vanilla on both branches, while clearly beating DML (+1.83/+2.81 on test). This is the current
  standard baseline — see `docs/notes/progress_log.md`'s "Standardizing on no-tech-flag" section for
  the full re-run detail and why the earlier (`is_4G`-included) numbers were superseded.

- [x] **Cleaned up experiment folder naming** — was `netztest_<mode>_rat_mlp_small_rat_mlp_large`
      (verbose, model names always the same so redundant); now `netztest_<mode>` (or
      `netztest_<mode>_withtechflag` for the legacy comparison mode). All old folders deleted
      (already committed to git, so nothing lost) and every run redone fresh under the new names.

## Next — the two things asked for previously (done, now under the old tech-flag setup — see above for current numbers)

### 1. Held-out test-set evaluation — DONE
- [x] `src/eval_netztest.py` written: loads saved checkpoints, evaluates on `test_df`, reports
      accuracy per branch per mode alongside the validation numbers.
- [x] Test numbers confirm the validation-based story: DML still clearly worst, tuned KDCL edges
      past vanilla for both branches (3G: 67.66% vs 67.47%; 4G: 68.42% vs 68.37%). See
      `progress_log.md` Run 8 for the full table.

### 2. RAT-selection inference stage — DONE
- [x] `src/infer_netztest.py` written, confound fix applied from day one (forces `is_4G` to each
      branch's own native value before scoring — see `build_explanation.md`).
- [x] Evaluated on the held-out test split with a majority-class baseline. First result: the
      recommender was degenerate — recommended "4G" for 100% of test rows (66.1%/99.8% headline
      numbers were mechanical artifacts, not real reasoning).
- [x] **Fixed**: added `--exclude_tech_flag` to `train_netztest.py`, retrained without `is_4G` as a
      feature, forcing both branches to reason from the 15 genuine radio-condition features alone.
      Recommender is no longer degenerate (splits 2,420 vs 2,945 across the test set). Honest
      result: overall agreement (65.6%) sits at the majority baseline, but agreement specifically on
      rows that achieved a genuinely top-tier outcome is 80.2% — meaningfully above baseline, and
      earned through real per-row reasoning this time. See `progress_log.md` Run 10.

## Also outstanding (lower priority, not blocking the two above)

- [ ] Repeated-seed runs — everything so far is a single seed; the tuned-KDCL-vs-vanilla gap
      (under half a point on both branches, current baseline) is within plausible run-to-run noise
      and shouldn't be over-claimed without this.
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
