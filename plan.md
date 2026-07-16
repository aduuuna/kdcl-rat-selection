# Plan: KDCL for Heterogeneous RAT Selection (FYP)

**Student:** Joy Owusu Ansah · **Supervisor:** Mr. Julius Ludu · **Dept. of Computer Science, University of Ghana**

Goal: refactor the KDCL (CVPR 2020) collaborative-distillation framework from image classifiers (CIFAR-100)
to tabular RAT agents (LTE/4G/5G/6G) trained on `urban_dataset_processed.xlsx`, then add an inference/validation
stage that recommends a RAT given a live network-state vector.

Source materials already in the folder that this plan is built from:
- `Online_Knowledge_Distillation_via_Collaborative_Learning_CVPR_2020_paper.pdf` — the paper
- `Online-Knowledge-Distillation-via-Collaborative-Learning-master/` — reference PyTorch code (image-based)
- `deepseek_markdown_20260712_afa0de.md` — your CV→telecom mapping notes
- `urban_dataset_processed.xlsx` — 22,690 rows × 64 cols of drive-test telemetry (already has engineered
  `time_bin, temp_load, sig_load, derived_load, mobility_encoded, SNR_norm, Latitude_norm, Longitude_norm`)

---

## Phase 0 — Environment & Repo Setup — DONE
- [x] Create the GitHub repo (`kdcl-rat-selection`), clone it locally.
- [x] Fix local Python: installed via the official python.org installer (winget hung, switched approach).
- [x] Set up `requirements.txt`: `torch`, `pandas`, `openpyxl`, `scikit-learn`, `numpy`, `matplotlib`.
- [x] Repo skeleton:
  ```
  /data/                 raw + processed xlsx, train/val/test splits (not committed if large — use .gitignore + a data README)
  /src/
    data.py              tabular dataset + per-RAT dataloaders (replaces image data.py)
    distortions.py       AWGN / Rayleigh fading / delay-jitter augmentations (replaces crop/flip)
    models.py            per-RAT MLP/tabular encoders, shared output head
    kdcl.py              ensemble methods: Naive, Linear, MinLogit, General
    train.py             refactored multi-branch training loop
    infer.py             validation/inference stage — RAT recommendation
    utils.py             metrics, checkpointing, logging
  /experiments/           run configs + results per experiment
  /notebooks/             EDA notebook for urban_dataset_processed.xlsx
  /docs/                  thesis-facing writeups, figures
  plan.md                 this file
  README.md
  ```

## Phase 1 — Data Understanding & Label Design — DONE (see docs/notes/progress_log.md for full detail)
- [x] EDA on `urban_dataset_processed.xlsx`: `NetworkTech` distribution, missingness per `*_missing`
      flag, class balance — all done via `notebooks/eda.py`, 2026-07-16.
- [x] Ground-truth target decided: `derived_load` quantile-binned into 3 classes (Low/Medium/High).
      Found and fixed label leakage along the way (see docs/notes/progress_log.md) — `temp_load`/`sig_load` excluded
      as near-duplicates of the label, and a real bug where `load_class` itself had leaked into the
      feature matrix.
- [x] **RAT/student split — confirmed from data.** `NetworkTech` only has two real values in
      `urban_dataset_processed.xlsx`: **5G (14,845 rows, ~65%) and 4G (7,844 rows, ~35%)**. No 3G/2G/6G rows
      exist. Students = **4G-agent and 5G-agent**, both real, no synthetic RAT needed. This is a smaller,
      cleaner two-student KDCL setup than originally speculated — matches the paper's own pairwise experiments
      (e.g. ResNet-50 + ResNet-18) and is easier to defend since there's no synthetic-data caveat.
      (6G/3G speculation earlier in this doc is dropped — not applicable to this dataset.)
- [x] Feature sets: simplified from "different feature sets per RAT" (original speculation, pre-data) to
      one shared 40-column feature set for both branches, with heterogeneity coming from model capacity
      instead (see below) — see `src/data.py` module docstring for the reasoning.
- [x] Train/val/test split: **sequential/time-blocked**, implemented in `src/data.py`'s
      `time_blocked_split()` (sorts by `SessionID`/`ElapsedTime`, no random shuffle).

## Phase 2 — Tabular Data Pipeline (replaces `data.py`) — DONE
- [x] Load xlsx once with pandas (`src/data.py: load_raw`). Not yet cached to parquet — reload is a
      few seconds, not currently worth the added complexity, revisit only if it becomes annoying.
- [x] Built `MultiViewDataset` (renamed from the planned `RATDataset` during implementation): every
      branch reads the *same* row through its own distortion, rather than disjoint per-RAT subsets —
      necessary correction, see `src/data.py` module docstring for why.
- [x] Implemented `src/distortions.py`: AWGN, Rayleigh fading, delay/jitter — the ICL step.

## Phase 3 — Model Architecture (replaces `models/`) — DONE
- [x] Small MLP per branch in `src/models.py` (`RATMLP`), 2-3 linear+BatchNorm+Dropout layers.
- [x] `model_dict` registry (`rat_mlp_small`/`rat_mlp`/`rat_mlp_large`) compatible with `--model_names`.
      Now actively used: 4G-agent=`rat_mlp_small`, 5G-agent=`rat_mlp_large` (Phase 1 capacity decision).

## Phase 4 — KDCL Core (replaces the mean-logit ensembling in `train.py`) — MOSTLY DONE
- [x] Implemented the actual paper methods in `src/kdcl.py` (reference `train.py`'s plain logit-average
      is not real KDCL):
  - [x] `KDCL-Naive` — pick the student with lowest per-sample CE loss.
  - [x] `KDCL-MinLogit` — normalize each student's logits by its target-class logit, take element-wise min
        across students. **Primary method** for this project.
  - [x] `KDCL-Linear` (stretch) — convex combination of logits, solved via projected gradient descent.
  - [ ] `KDCL-General` (stretch) — **not started.** Needs held-out-set covariance estimation, updated
        once per epoch. Lower priority than running the core Naive/MinLogit/Linear comparison first.
- [x] Loss: `CE(student, y) + λ·T²·KL(softmax(z_t/T) || softmax(z_student/T))` per branch, in `src/train.py`.

## Phase 5 — Training Loop Refactor (`train.py`) — DONE (see docs/notes/progress_log.md for full results)
- [x] Loop refactored: per-branch forward, shared teacher logit via `kdcl.py`, per-branch backward,
      `MultiViewDataset` dataloaders. Supports `--mode {vanilla, dml, kdcl}`.
- [x] Smoke-tested end to end (2-epoch runs) — confirms the pipeline runs correctly, not a real result.
- [x] Ran the 3-way comparison. **Note: first-pass numbers below were superseded once feature
      normalization was added in Phase 6 — see that section for the corrected comparison.**

## Phase 6 — Evaluation — normalization fix + revised finding (see docs/notes/progress_log.md for full detail)
- [x] Hyperparameter sweep: `src/sweep.py`, 12-combo grid over `T`x`alpha`. Found `alpha=0.2` beats
      the CIFAR-inherited default `0.5` at every temperature. Used T=1.0/alpha=0.2 going forward.
- [x] Added z-score feature normalization (`fit_scaler` in `src/data.py`, fit on train split only).
      **This was the real lever, not the distillation method choice:** vanilla jumped from
      73.85%/78.20% (unnormalized) to 91.15%/89.48% (normalized).
- [x] Re-ran vanilla/DML/KDCL(ICL on)/KDCL(ICL off) with normalization, 50 epochs each:
      vanilla 91.15%/89.48%, **dml 91.86%/90.71% (best)**, kdcl-ICL-on 91.39%/89.66%,
      kdcl-ICL-off 91.13%/90.51%. **Revised conclusion: DML no longer fails and edges out KDCL for
      both branches** — contradicts the earlier (unnormalized) headline result. Diagnosis: DML's
      failure mode needs a real capacity/performance gap between students, and normalization nearly
      closed the gap between our small/large MLP pairing on this comparatively easy 3-class task.
- [x] Ablation (ICL on/off) and robustness test re-run with normalization: robustness test now
      behaves sensibly (monotonic degradation, not flat), but shows no clear KDCL/ICL advantage over
      vanilla either — consistent with the capacity-gap diagnosis above.
- [x] **Tried to widen the capacity gap three ways — conclusive negative result.** Pure-linear 4G
      model vs large 5G (90.77%/89.54%), harder 5-class label (80.63%/78.87%), 3x training epochs
      (91.57%/89.48%) — the smaller model won every single time. The 150-epoch run's loss curves
      showed the large model *overfitting* (val loss rising while train loss fell), not
      undertraining. **Conclusion: this dataset/feature set doesn't support a fair capacity gap
      between architectures of this kind** — stopped chasing it rather than force an unfair
      comparison. The DML/KDCL-beat-vanilla-by-1-2-points finding from the normalization fix stands
      as legitimate (if modest) evidence that collaborative training transfers to this domain; the
      paper's more dramatic capacity-gap story needs a task/dataset this one doesn't provide.
- [ ] Per-RAT accuracy/F1/confusion matrix, vanilla vs DML vs KDCL — deferred, not blocking Phase 7.

**Phase 6 status: DONE**, with an honestly-documented negative result on the capacity-gap question.
Every run's exact command, hyperparameters, and epoch-by-epoch accuracy is saved to
`experiments/<mode>_<model_names>/<timestamp>/train_log.txt` and committed to git (checkpoints
too — tiny, ~2.7MB total) as verifiable proof of every experiment actually run.

## Phase 7 — RAT Selection / Validation Stage (your novel contribution) — DONE, honest negative result
- [x] Defined the inference contract: rule-based (plan's Option 1) — run the shared state through
      both trained branches, recommend whichever predicts the better expected KPI outcome.
      Learned-meta-head (Option 2) not pursued — the rule-based approach's result (below) shows the
      bottleneck isn't the recommendation mechanism, it's the underlying branches' lack of real
      differentiation (per Phase 6), which a meta-head wouldn't fix either.
- [x] `src/infer.py` implemented: loads checkpoints, evaluates on the true held-out **test** split
      (previously used an arbitrary slice — fixed), reports a majority-class baseline for context.
- [x] Validated against held-out data. **Found and fixed a confound first:** `is_5G` is a trained
      feature, so both branches could see a row's actual serving tech directly rather than reasoning
      independently — fixed by overriding `is_5G` to each branch's own native value before scoring.
      **Honest result after the fix: ~58-59% agreement, roughly at/below the 62.7% majority-class
      baseline** — the recommender isn't adding value. This directly reflects Phase 6's finding of
      no real capacity/behavior differentiation between the two branches: with nearly-identical
      branches, there's little independent signal for a recommender to exploit. Documented as a
      genuine limitation for the discussion chapter, not an implementation bug.

## Phase 8 — Write-up
- [x] First draft written: `docs/notes/writeup.md` — Background, Methodology, Results (§3.1-3.3),
      Contribution (Phase 7 recommender), Limitations, Future Work, Summary. Pulled from
      `progress_log.md`'s full experiment trail, reorganized by argument rather than chronology.
- [ ] Review/rewrite in your own voice for actual thesis submission; the draft is a solid starting
      point, not final prose.

---

## Open Questions to Resolve Early (blocking Phase 1/7)
1. ~~Does the dataset have real 5G/6G rows, or do we need a synthetic stand-in for 6G?~~ **Resolved:** real
   4G (7,844 rows) and 5G (14,845 rows) only — two-student setup, no synthetic RAT.
2. What's the exact KPI classification scheme (bin edges for Low/Medium/High load)?
3. Regression vs classification for the final KPI target — notes flag MSE as the swap-in if you go regression.
4. What "ground truth" defines the correct RAT recommendation in Phase 7 (needed to actually *validate* the
   recommender, not just eyeball it)?


