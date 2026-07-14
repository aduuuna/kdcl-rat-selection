# Plan: KDCL for Heterogeneous RAT Selection (FYP)

**Student:** Kofi · **Supervisor:** Mr. Julius Ludu · **Dept. of Computer Science, University of Ghana**

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

## Phase 0 — Environment & Repo Setup
- [ ] Create the GitHub repo (name suggestion below), clone it locally.
- [ ] Fix local Python: the `python`/`python3` on this machine currently resolve to the Windows Store stub
      (no real interpreter found). Install a real Python 3.10/3.11 from python.org (or via `winget`/`conda`)
      before anything else will run.
- [ ] Set up `requirements.txt`: `torch`, `pandas`, `openpyxl`, `scikit-learn`, `numpy`, `matplotlib`.
- [ ] Repo skeleton:
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

## Phase 1 — Data Understanding & Label Design
- [ ] EDA notebook on `urban_dataset_processed.xlsx`: distribution of `NetworkTech` (2G/3G/4G/5G), missingness
      per `*_missing` flag, class balance for candidate labels.
- [ ] Decide the **ground-truth target**: discretize `derived_load` (or `DL_bitrate`) into classes
      (e.g. Low/Medium/High, matching the scheme in your notes) — confirm bin edges make sense per RAT.
- [x] **RAT/student split — confirmed from data.** `NetworkTech` only has two real values in
      `urban_dataset_processed.xlsx`: **5G (14,845 rows, ~65%) and 4G (7,844 rows, ~35%)**. No 3G/2G/6G rows
      exist. Students = **4G-agent and 5G-agent**, both real, no synthetic RAT needed. This is a smaller,
      cleaner two-student KDCL setup than originally speculated — matches the paper's own pairwise experiments
      (e.g. ResNet-50 + ResNet-18) and is easier to defend since there's no synthetic-data caveat.
      (6G/3G speculation earlier in this doc is dropped — not applicable to this dataset.)
- [ ] Define per-RAT feature sets (per your notes: LTE = small feature set, 5G = + beamforming/CQI-style
      features, 6G = + extra synthetic features) drawn from the 64 available columns, all mapped to the
      **same output class space**.
- [ ] Train/val/test split: **sequential/time-blocked**, not random shuffle (data is temporally correlated —
      use `SessionID`/`Timestamp`/`ElapsedTime` to keep consecutive samples together, matching your notes on
      fixing Eq. 10 for wireless data).

## Phase 2 — Tabular Data Pipeline (replaces `data.py`)
- [ ] Load xlsx once with pandas, cache to parquet/pickle for fast reloads (43MB xlsx sheet is slow to re-parse).
- [ ] Build a `RATDataset(torch.utils.data.Dataset)` that, for a given base row, returns one tensor per branch
      (mirrors the `img[:, model_idx, ...]` pattern in the current `train.py`, but for feature vectors).
- [ ] Implement `distortions.py`: AWGN, Rayleigh fading multiplier, random delay/jitter injection — applied
      per-branch to the same base row (this is your ICL/Invariant Collaborative Learning step, replacing
      `RandomCrop`/`RandomHorizontalFlip`).

## Phase 3 — Model Architecture (replaces `models/`)
- [ ] Small MLP per RAT (input dim = that RAT's feature count → shared hidden layers → shared output logit
      dimension = number of KPI classes). Start simple (2-3 linear layers + BatchNorm/Dropout); this is tabular
      data, not images, so nothing here needs to be as deep as ResNet/WRN.
- [ ] Keep `model_dict`-style registry so `train.py` can still take `--model_names` style args, just pointing
      at the new tabular models instead of `resnet20` etc.

## Phase 4 — KDCL Core (replaces the mean-logit ensembling in `train.py`)
- [ ] **Note:** the current reference `train.py` only averages logits (`outputs.mean(dim=0)`) — that is
      *not* MinLogit, it's closer to a naive online ensemble. Implement the actual paper methods in `kdcl.py`:
  - [ ] `KDCL-Naive` — pick the student with lowest per-sample CE loss.
  - [ ] `KDCL-MinLogit` — normalize each student's logits by its target-class logit, take element-wise min
        across students. **Primary method** — matches your notes' recommendation for the LTE/5G/6G gap problem.
  - [ ] `KDCL-Linear` (stretch) — convex combination of logits solved per batch.
  - [ ] `KDCL-General` (stretch) — inverse-covariance weighting on a held-out validation block, updated once
        per epoch, using **sequential blocks** for the correlation matrix (per your notes' Eq. 10 fix).
- [ ] Loss: `L = CE(student, y) + λ · T² · KL(softmax(z_t/T) || softmax(z_student/T))` per branch, same as `train.py`
      but pointed at the new ensemble function.

## Phase 5 — Training Loop Refactor (`train.py`)
- [ ] Reuse the existing loop structure (per-branch forward, shared teacher logit, per-branch backward) but:
  - swap image batches for per-RAT feature tensors,
  - swap the ensemble step for `kdcl.py`'s MinLogit,
  - swap CIFAR100 dataloaders for `RATDataset`.
- [ ] Baselines to run for comparison (needed for your results chapter):
  1. Each RAT model trained **alone** (vanilla, no distillation).
  2. Each RAT model trained with **DML** (mutual KL between peers, no ensemble) — to reproduce the
     "weak model hurts strong model" failure mode your notes describe.
  3. Each RAT model trained with **KDCL-MinLogit** — should beat both of the above, especially for the
     stronger RAT(s).

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

---

## Suggested GitHub repo name
**`kdcl-rat-selection`**

Alternatives if you want it more descriptive or more branded to the course: `heterogeneous-rat-kdcl`,
`wireless-kdcl-fyp`. Keep it lowercase-hyphenated (GitHub convention), and keep "KDCL" in the name since
that's the core technique a reader/examiner will recognize.
