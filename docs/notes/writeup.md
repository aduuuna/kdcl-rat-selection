# Project Write-Up: KDCL for Heterogeneous RAT Selection

This maps the project's work into a thesis-chapter structure (per `plan.md` Phase 8). It's written
from `docs/notes/progress_log.md`'s full experiment trail, reorganized by argument rather than by
chronology, so it reads as a coherent narrative rather than a day-by-day log. Treat this as a solid
first draft to react to and rewrite in your own voice, not final thesis prose.

---

## 1. Background

**The problem KDCL solves.** Training a small, efficient neural network to perform as well as a
large one traditionally requires *offline knowledge distillation*: train a large "teacher" model to
convergence first, then use its outputs to supervise a smaller "student." This is expensive (the
teacher must be fully trained before the student can even start) and one-directional (the student
never improves the teacher). Online alternatives like Deep Mutual Learning (DML) let multiple models
train together and mimic each other's predictions, removing the need for a pre-trained teacher — but
DML has a specific, well-documented failure mode: when there is a real performance gap between the
models being trained together, the weaker model's noisy predictions can drag the stronger model
down, actively hurting it relative to training alone.

**KDCL's core idea** (Guo et al., CVPR 2020) is to replace pairwise mimicry with a *dynamically
generated ensemble teacher*: at every training step, the logits from all student models are combined
into a single soft target, and every student is trained against both the ground-truth label and this
shared soft target. The paper's key technical contribution is **MinLogit**: normalize each student's
logits against its own target-class logit, then take the element-wise minimum across students. This
produces a teacher signal that is mathematically guaranteed to be no worse than any single student on
the ground-truth class — so a weak student can contribute useful signal without ever dragging a
strong student down. The paper demonstrates this on CIFAR-100 and ImageNet: pairings like
ResNet-50/ResNet-18 or ResNet-50/MobileNetV2x0.5 both improve under KDCL, including cases where a
much smaller model measurably helps a much larger one.

**Why this seemed relevant to wireless RAT selection.** Different Radio Access Technologies (RATs) —
4G/LTE vs 5G, in this project's case — have inherent performance differences (different spectral
efficiency, latency, coverage characteristics), which looked analogous to the capacity gaps KDCL was
designed to handle. The motivating hypothesis was: train a "4G-agent" and "5G-agent" model
collaboratively via KDCL, using each other's predictions to improve, in the same way a small and
large vision model improve each other in the original paper. The dataset available for this
(`urban_dataset_processed.xlsx`) is real drive-test telemetry — 22,690 rows, confirmed to contain
only 4G (7,844 rows) and 5G (14,845 rows) samples, no 3G or 6G. This grounded the plan in real data
rather than the originally-speculated 3-or-4-RAT hypothetical.

---

## 2. Methodology

### 2.1 Adapting KDCL from images to tabular network data

The reference implementation (a public CIFAR-100 reimplementation of the paper) trains multiple
image classifiers on the *same* image, each seeing a differently-augmented view (random crop/flip),
and ensembles their predictions into one soft target. The key design decision for this project was
recognizing that **the RAT "students" must follow the same pattern**: every training instance is one
shared network-state row, and each branch (4G-agent, 5G-agent) sees it through its own simulated
channel distortion, not a disjoint subset of rows. This mattered because RAT tech in this dataset is
never observed twice for the same moment (a row is measured under either 4G or 5G, never both) — so
naively splitting the dataset by `NetworkTech` into two "each RAT trains on its own rows" subsets
would break the mechanism that makes a shared ensembled teacher logit meaningful in the first place.
This is implemented in `src/data.py`'s `MultiViewDataset`.

### 2.2 Label design

The target is `derived_load`, an engineered continuous KPI already present in the dataset,
quantile-binned into 3 classes (Low/Medium/High) via `pd.qcut` — chosen over regression to keep the
task classification-based, matching the paper's setup, with room to switch to MSE-based regression
later if needed (noted as an open option, not pursued).

**Two label-leakage issues were found and fixed during EDA** (`notebooks/eda.py`):
- `temp_load` and `sig_load` correlate with `derived_load` at 0.79 and 0.66 respectively; `sig_load`
  additionally correlates with `SNR` at exactly the mirror value (-0.665), indicating it's a rescaled
  transform of `SNR`, not independent information. Both were excluded from the feature set as likely
  components of the `derived_load` formula itself.
- A more serious bug: the discretized label (`load_class`) was briefly present *inside* the feature
  matrix due to a function-call-ordering issue (`numeric_feature_columns()` excluded the raw
  `derived_load` column by name but not the label derived from it, and the label was added to the
  dataframe before that function was called). Confirmed via exact column-count arithmetic and fixed
  by making the exclusion order-independent.

Final feature set: 40 numeric columns, covering signal quality (SNR, CQI, Level, RSRP-style
metrics), throughput (DL/UL_bitrate), mobility, geolocation, neighbor-cell metrics, and their
`*_missing` indicator flags (the dataset ships explicit missingness flags for several neighbor-cell
fields with 20-38% missingness; these flags are kept as features so the model can distinguish
"missing, imputed to 0" from "genuinely 0" rather than dropping those columns).

### 2.3 Train/val/test split

Sequential/time-blocked (`time_blocked_split`, sorted by `SessionID`/`ElapsedTime`), not a random
shuffle — wireless telemetry is temporally correlated, so a random split would leak adjacent,
near-identical samples across train/val/test.

### 2.4 Model architecture

Small MLPs (`src/models.py`) rather than anything CNN-like, appropriate for tabular input:
`rat_mlp_small` (64→32 hidden units), `rat_mlp` (128→64), `rat_mlp_large` (256→128→64), and
`rat_mlp_tiny` (a single linear layer, added later for the capacity-gap investigation, see §3.4). A
shared `model_dict` registry lets any pair of these be selected per branch via CLI, mirroring the
paper's use of differently-sized architectures for different students.

### 2.5 KDCL ensemble methods

`src/kdcl.py` implements the paper's actual methods — **the reference CIFAR implementation this
project started from only averages logits, which is not any of the paper's four methods**:
- `naive`: per-sample, pick whichever student has lowest cross-entropy loss.
- `minlogit`: the paper's primary contribution — described above, this project's main method.
- `linear`: convex combination of student logits minimizing training loss, solved via projected
  gradient descent (a lightweight substitute for a full QP solver, adequate for 2 students).
- `general` (inverse-covariance weighting on a held-out block) was scoped as a stretch goal and not
  implemented, given time constraints and that it adds meaningfully more complexity for uncertain
  benefit with only 2 students.

### 2.6 Training loop and comparison modes

`src/train.py` supports three modes for the same underlying loop, differing only in what supervises
each branch: **vanilla** (ground-truth cross-entropy only, no distillation — the baseline), **dml**
(mutual KL divergence between the two branches' softmax outputs, no ensembling — reproduces the
paper's failure-mode setup), and **kdcl** (ensemble teacher logit via `kdcl.py`, default MinLogit).
Distortions (`src/distortions.py`: AWGN noise, Rayleigh fading, delay/jitter) are applied per-branch
during training as the tabular analogue of the paper's Invariant Collaborative Learning (ICL) step.

---

## 3. Results

### 3.1 Initial comparison (unnormalized features) — a promising but ultimately misleading result

First full 50-epoch runs, default hyperparameters (T=2.0, alpha=0.5, `rat_mlp_small`/`rat_mlp_large`
for 4G/5G):

| Mode    | 4G_agent | 5G_agent |
|---------|----------|----------|
| vanilla | 73.85%   | 78.20%   |
| dml     | 64.85%   | 64.83%   |
| kdcl    | 64.80%   | 68.44%   |

DML clearly reproduced the paper's negative-transfer failure — both branches collapsed toward the
same value, the stronger 5G branch dragged down by the weaker 4G branch. KDCL clearly beat DML
(especially protecting 5G) but had not yet beaten vanilla.

**Hyperparameter sweep** (`src/sweep.py`, 12-combo grid over `T`x`alpha`) found `alpha=0.2` decisively
beats the CIFAR-inherited default of `0.5` at every temperature tested — with only 3 output classes
(vs. 1000-way ImageNet), the soft-target distribution carries much less information, so leaning
harder on ground-truth loss works better than in the original paper's setting. Confirmed at full
50 epochs (T=1.0, alpha=0.2): **5G_agent reached 78.72%, beating vanilla's 78.20%** — at the time,
this looked like the paper's headline claim reproduced.

### 3.2 A methodological correction that changed the conclusion

Two follow-up checks (an ICL ablation, and a noise-robustness test mirroring the paper's Figure 4)
turned up the same root cause: the distortions used a **fixed absolute noise magnitude**, but the 40
features span wildly different raw scales (e.g. `Level` ~ -100, `DL_bitrate` ~ hundreds, vs. the
already-normalized `*_norm` columns at 0-1). A fixed perturbation is negligible for large-scale
features and disproportionate for normalized ones. Symptom: disabling the distortion *improved*
accuracy (opposite of the paper's expected small drop), and the robustness test showed flat accuracy
across all injected noise levels for every mode — not evidence of robustness, evidence the test
wasn't stressing anything.

**Fix:** added z-score feature normalization (`fit_scaler` in `src/data.py`, fit on the training
split only, applied consistently to train/val). This was, by a wide margin, **the single biggest
lever in the whole project** — bigger than any distillation-method choice:

| Config                           | 4G_agent | 5G_agent |
|-----------------------------------|----------|----------|
| vanilla                           | 91.15%   | 89.48%   |
| dml                                | **91.86%** | **90.71%** |
| kdcl (T=1.0, alpha=0.2, ICL on)   | 91.39%   | 89.66%   |
| kdcl (T=1.0, alpha=0.2, ICL off)  | 91.13%   | 90.51%   |

Vanilla alone jumped ~13-17 points once features were properly scaled — confirming the earlier
"KDCL beats vanilla" result was measured on a poorly-conditioned pipeline. With clean inputs, **DML
no longer fails at all and now slightly outperforms KDCL for both branches**, directly reversing the
earlier headline finding.

### 3.3 Diagnosing why DML stopped failing: no real capacity gap

DML's failure mode specifically requires a real, sizeable performance gap between the collaborating
models. The paper's own pairings have one by construction (e.g. ResNet-50 at 76.8% vs. ResNet-18 at
71.2%, a hard 1000-way task). Once normalized, this project's own vanilla gap between the small and
large MLP had nearly vanished (91.15% vs. 89.48% — under 2 points, and the *smaller* model was
ahead). With no real gap for DML to exploit, there's nothing for KDCL's mechanism to visibly protect
against either.

Three separate attempts were made to deliberately widen this gap, all under vanilla training so the
comparison stays clean:

| Attempt                                                    | Weaker branch | Stronger branch |
|--------------------------------------------------------------|-----------------|--------------------|
| 3-class label, small vs. large MLP                            | 91.15%          | 89.48%             |
| 3-class label, **pure-linear model** (no hidden layer) vs. large MLP | 90.77%    | 89.54%             |
| **5-class label** (harder task), small vs. large MLP           | 80.63%          | 78.87%             |
| 3-class label, small vs. large MLP, **150 epochs** (3x training)  | 91.57%          | 89.48%             |

The smaller/weaker architecture won in every single attempt. The 150-epoch run's training curves
were diagnostic: the large model's training accuracy kept climbing while its *validation loss also
climbed* — a textbook overfitting signature, not undertraining. **Conclusion: this dataset's feature
set and label design do not support a fair capacity gap between MLP architectures of the kind tried
here.** Forcing a bigger nominal gap (an even deeper model) would only measure worsening overfitting,
not a genuine capability difference, so this line of investigation was stopped deliberately rather
than continued indefinitely.

**What still stands:** both DML and KDCL give a small, consistent edge over vanilla training even
without a capacity gap (~1-2 points across the normalized runs) — legitimate, if modest, evidence
that collaborative multi-branch training transfers to this tabular RAT domain without breaking
anything. The paper's more dramatic "weak model rescues/threatens strong model" story specifically
requires a capacity gap that this dataset, with MLP-style architectures, does not appear to provide.

---

## 4. Contribution: The RAT-Selection Inference Stage

Beyond reproducing KDCL's training mechanism, this project's own contribution is a downstream
**RAT-selection recommender** (`src/infer.py`): given a network-state vector, run it through both
trained branches and recommend whichever predicts the better expected KPI outcome (a rule-based
approach, chosen over a learned meta-head given the underlying branches' limited differentiation —
see below).

Evaluated properly on the true held-out test split (previously an oversight — an early version used
an arbitrary slice of the full dataframe instead), with a majority-class baseline added for context.
**A confound was found and fixed first:** `is_5G` (whether a row was actually served by 5G) is one
of the 40 trained features, so both branches could see a row's real serving tech directly rather
than reasoning independently about "what would my tech's model predict for this state." Fixed by
overriding `is_5G` to each branch's own native identity before scoring.

| Metric                                                    | Before fix | After fix |
|--------------------------------------------------------------|--------------|-------------|
| Overall agreement with actual serving RAT                    | 60.3%        | 58.5%       |
| Majority-class baseline                                       | 62.7%        | 62.7%       |
| Agreement on rows that actually achieved the top load class  | 79.9%        | 59.3%       |

**Honest result: once the confound is removed, the recommender performs roughly at or below
baseline.** This is not an implementation bug — it directly reflects §3.3's finding that the two
trained branches don't behave meaningfully differently from each other. A recommender comparing two
near-identical models has little genuine signal to work with. This is a legitimate, reportable
limitation rather than a result to hide.

---

## 5. Limitations

- **No real capacity/performance gap achievable** between the two RAT branches on this dataset with
  MLP-style architectures, across three separate attempts (architecture size, label granularity,
  training budget) — the central mechanism KDCL is designed around could not be fairly exercised.
- **The RAT-selection recommender inherits this limitation**: it cannot meaningfully differentiate
  between two branches that don't meaningfully differentiate between themselves.
- **`KDCL-General`** (the paper's fourth ensemble variant, using held-out-set covariance estimation)
  was not implemented — scoped as a stretch goal, deprioritized in favor of running the core
  Naive/MinLogit/Linear comparison properly.
- **Single-seed experiments throughout** — no repeated trials or error bars, so the small (1-2 point)
  differences between vanilla/DML/KDCL in the normalized comparison should be read as suggestive,
  not statistically confirmed.
- **Missingness handling is basic**: `fillna(0.0)` plus the dataset's own `*_missing` indicator
  flags as features, rather than a more sophisticated imputation scheme. Considered adequate for
  this project's scope but noted as a simplification.
- **Two real, distinct bugs were found and fixed during the project** (label leakage into features,
  and a fixed-magnitude-distortion vs. feature-scale mismatch) — evidence of a genuine debugging
  process, but also a reminder that earlier "positive" results (pre-fix) were not trustworthy and
  had to be revised rather than reported as-is.

## 6. Future Work

- **Try a harder or continuous target**: regression on the raw KPI (swap KL-divergence for MSE, as
  originally scoped) or a finer-grained classification task might better expose real
  capacity-dependent behavior than the current 3-class label, which turned out to be
  near-linearly-separable (a pure linear model matched a deep MLP).
- **Genuinely different feature subsets per RAT** (matching the original design sketch: LTE gets a
  smaller feature set, 5G gets additional metrics) rather than only varying model depth/width, as a
  different way to create real behavioral heterogeneity between branches.
- **Repeated-seed runs** to put error bars on the vanilla/DML/KDCL comparison and confirm which
  differences are real vs. noise.
- **A learned meta-head** for RAT selection (plan's Option 2), conditioned on more informative or
  differentiated branch outputs — likely only worthwhile after addressing the capacity-gap
  limitation above, since a meta-head can't extract signal the branches don't have.
- **Scaling to more RATs** (6G/7G) if/when comparable real telemetry becomes available, extending the
  same shared-instance/multi-branch design beyond the current two-student setup.

## 7. Summary

The core question — can KDCL's collaborative distillation mechanism be transferred from image
classifiers to heterogeneous wireless RAT models — has a qualified, honestly-earned answer: **yes,
mechanically and without breaking anything (both DML and KDCL give a small, consistent edge over
standalone training), but the paper's more dramatic result (a real performance gap where a weak
model measurably helps, and doesn't hurt, a strong one) could not be reproduced on this dataset**,
because the dataset and label design don't support a genuine capacity gap between the RAT-agent
architectures tried, confirmed through three separate, deliberate attempts. This conclusion was
reached through a real methodological process — catching label leakage, a feature-scale bug, and a
recommendation-stage confound along the way — which is itself worth presenting as evidence of rigor,
not just the final numbers.
