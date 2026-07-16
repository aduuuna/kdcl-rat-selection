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
- [x] Ran the 3-way comparison, 50 epochs each: vanilla (73.85%/78.20%), DML (64.85%/64.83% —
      reproduces the paper's negative-transfer failure), KDCL-MinLogit (64.80%/68.44% — clearly
      beats DML, does not yet beat vanilla). `T`/`alpha` tuning needed next — see Phase 6.

## Phase 6 — Evaluation
- [ ] Per-RAT accuracy/F1/confusion matrix, vanilla vs DML vs KDCL.
- [ ] Ablation: with/without the distortion (ICL) step — expect a small but real accuracy drop when disabled,
      mirroring the paper's ~0.5% finding.
- [ ] Robustness test: inject increasing noise (larger AWGN variance) into the validation set and plot
      accuracy/loss degradation for KDCL vs vanilla vs DML (mirrors the paper's Figure 4).
- [ ] Hyperparameter sweep: temperature `T` and loss weight `λ` (start from the paper's stable range 0.2–5,
      default λ=1).

## Phase 7 — RAT Selection / Validation Stage (your novel contribution)
- [ ] Define the inference contract: given a live network-state vector (whatever RAT(s) it has sensors for),
      run it through the trained branch(es) and produce a **recommendation of which RAT to select/handover to**
      — not just a KPI class.
- [ ] Design options to decide between (flag for supervisor input):
  1. **Rule-based on predicted KPI**: run the same state through each RAT's trained head (if features are
     available/estimable for each), compare predicted class/expected throughput, recommend the argmax.
  2. **Learned meta-head**: a small classifier trained on top of the ensemble's outputs whose target label
     is "which RAT historically performed best in this state" — needs a derived ground-truth for "best RAT
     per row" from the dataset (e.g. compare `DL_bitrate` across co-located `NetworkTech` samples).
- [ ] Implement `infer.py`: load trained checkpoints, accept a new sample, output the recommended RAT +
      confidence.
- [ ] Validate against a held-out slice: does the recommended RAT actually match/beat the RAT that was
      really in use for that sample (using `NetworkTech` as ground truth of what was actually selected)?

## Phase 8 — Write-up
- [ ] Map each phase's results into thesis chapters: Background (paper), Methodology (Phases 1-5), Results
      (Phase 6), Contribution/Discussion (Phase 7), Limitations & Future Work (7G scalability, per your slides).

---

## Open Questions to Resolve Early (blocking Phase 1/7)
1. ~~Does the dataset have real 5G/6G rows, or do we need a synthetic stand-in for 6G?~~ **Resolved:** real
   4G (7,844 rows) and 5G (14,845 rows) only — two-student setup, no synthetic RAT.
2. What's the exact KPI classification scheme (bin edges for Low/Medium/High load)?
3. Regression vs classification for the final KPI target — notes flag MSE as the swap-in if you go regression.
4. What "ground truth" defines the correct RAT recommendation in Phase 7 (needed to actually *validate* the
   recommender, not just eyeball it)?


