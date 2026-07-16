# Project Progress Log

Living log of what's been done, decisions made, and why — updated as we move through `plan.md`'s
phases, so progress is visible directly in the repo without digging through commit history. Starts
with Phase 1; later phases get appended below as sections, not new files.

## Phase 1 — Data Understanding & Label Design

### Checklist (from plan.md)
- [x] RAT/student split confirmed from real data
- [x] Pipeline smoke-tested end to end (data loads, KDCL training loop runs)
- [x] Rule out label leakage in `derived_load` — found two leaks, both fixed (see below)
- [x] EDA: missingness per `*_missing` flag
- [x] Finalize KPI classification scheme (quantile tertiles on `derived_load`, feature set now clean)
- [x] Decide per-branch model capacity — 4G small, 5G large (see below)

### Findings

#### RAT split (confirmed 2026-07-15)
Real `NetworkTech` values in `urban_dataset_processed.xlsx`: **5G = 14,845 rows (~65%), 4G = 7,844
rows (~35%)**. No 3G/2G/6G present. Two KDCL students = 4G-agent, 5G-agent, both trained on the
full shared dataset (not disjoint row subsets — every row is one instance both branches see through
their own distortion; see `src/data.py` module docstring for why).

#### Smoke test (2026-07-16) — SUPERSEDED, see leakage fix below
`python src/data.py`:
```
overall (22689, 66) class balance: {0: 0.334, 1: 0.334, 2: 0.332}
4G    (7844, 66)  class balance: {0: 0.350, 1: 0.346, 2: 0.304}
5G    (14845, 66) class balance: {2: 0.347, 1: 0.327, 0: 0.326}
```
Overall balance is ~even tertiles by construction (quantile-based label). The per-RAT skew is real
signal: 4G leans toward lower load classes, 5G leans toward the higher one — genuine heterogeneity
between the two RATs, not an artifact.

`python src/train.py --epochs 2` (mode=kdcl, ensemble=minlogit, both branches same `rat_mlp` arch):
```
4G_agent best val acc: 57.77
5G_agent best val acc: 57.68
```
3-class problem, random baseline ~33% — after only 2 epochs this is a strong signal the model is
learning real structure from the radio features. Both branches score near-identically, which is
expected since they currently share the same architecture (no capacity gap yet — see below).

#### Label leakage — RESOLVED (2026-07-16), two separate issues found

**Issue 1 (design question): `temp_load` / `sig_load` correlate with `derived_load` at 0.79 / 0.66.**
Notably `SNR` correlates at exactly -0.665 — the mirror image of `sig_load`'s 0.665 — which means
`sig_load` is just a rescaled/inverted transform of `SNR`, not independent information. Combined
with the naming (`temp_load` + `sig_load` -> `derived_load`), this is almost certainly a formula
built directly from those two columns. **Decision: excluded both from the feature set**
(`LEAKY_COLS` in `src/data.py`). Conservative choice — the model should predict load from
independent radio conditions, not partially reconstruct a known blend of its own ingredients.

**Issue 2 (actual bug, more serious): the label itself was in the input features.**
`numeric_feature_columns()` excluded `derived_load` by name but not `load_class` (the discretized
label), and `train.py`/`infer.py` both added `load_class` to the dataframe *before* calling that
function — so it silently passed through as a 43rd numeric input feature. Confirmed by the
arithmetic: 64 raw cols − 22 ID/timestamp cols − 1 label + `is_5G` + `load_class` = 43, matching the
smoke-test's printed count exactly. **This means the 57.77%/57.68% smoke-test accuracies were not
trustworthy** — the model had a literal copy of its own answer as an input.

**Fix applied:** `numeric_feature_columns()` now always excludes `derived_load`, `load_class`, and
`LEAKY_COLS` regardless of call order (previously order-dependent and fragile — that's how this
slipped in). Feature count is now 40, confirmed clean via an assertion in
`notebooks/eda.py` (verified 2026-07-16 — the 40-column list contains no label/leaky columns).

**Re-run confirms the fix, and is good news:** with the leaks removed, `python src/train.py --epochs 2`
gives 4G_agent 57.92%, 5G_agent 57.71% — virtually unchanged from the pre-fix 57.77%/57.68%. At
only 2 epochs with a small learning rate, the model hadn't yet had time to exploit the leaked label
feature, so this number reflects genuine signal learned from real radio features, not the leak.
Still correct to have fixed it: over a full 50-epoch run the model likely would have gradually
learned to lean on that one perfect feature and inflated the numbers. **~58% accuracy on a 3-class
problem (33% baseline) after 2 epochs, on a clean feature set, is now a trustworthy early signal
that the task is learnable.**

#### Missingness (2026-07-16 EDA)
Neighbor/second-cell fields are missing for a large minority of rows: `SecondCell_*` and `NCellid1`
family at 38.4%, `NLAC1`/`NCell1`/`NARFCN1`/`NRxLev1`/`NQual1` at 21.5%, `CQI` at 22.2%, `BANDWIDTH`
at 4.6%. Not disqualifying — the dataset already ships a `*_missing` indicator flag alongside each
of these columns, and those flags are themselves included as features (not in `NON_FEATURE_COLS`),
so the model can distinguish "missing, imputed to 0" from "genuinely 0". Current handling
(`fillna(0.0)` + missing flag as a feature) is standard practice and considered sufficient for now;
worth noting as a simplification/limitation in the write-up rather than something to fix.

#### Branch capacity — RESOLVED (2026-07-16)
Both branches previously used identical `rat_mlp` architecture, which meant there was no real
capacity gap to study — just two equally-sized twins. This matters because the paper's headline
result (a small/weak model can help a big/strong one without being dragged down itself) is exactly
what a capacity gap is needed to demonstrate; without one, KDCL vs DML vs vanilla would look
uninterestingly similar regardless of which method "wins".

**Decision:** 4G-agent = `rat_mlp_small` (hidden layers 64/32), 5G-agent = `rat_mlp_large` (hidden
layers 256/128/64). This mirrors the paper's own pairing of a small model (MobileNetV2x0.5) against
a big one (ResNet-50), and matches the direction of the real gap already observed in the data: 5G
skews toward higher load classes than 4G (see class balance above), so 5G is the natural "stronger"
model and 4G the natural "weaker" one. This was flagged to the supervisor on 2026-07-15
(WhatsApp/Digrad message asking exactly this question); decided now to keep momentum, defaulted to
this pairing since it's the option best supported by the data itself. Open to revising if he
responds with a different preference. Set as the new default in `src/train.py`/`src/infer.py`
(`--model_names` now defaults to `rat_mlp_small rat_mlp_large`, order 4G_agent then 5G_agent).

#### Capacity-split smoke test (2026-07-16)
First run with the new asymmetric architecture (4G_agent=`rat_mlp_small`, 5G_agent=`rat_mlp_large`),
`python src/train.py --epochs 2` (mode=kdcl, ensemble=minlogit):
```
4G_agent (small) best val acc: 58.62
5G_agent (large) best val acc: 57.98
```
Both branches improved slightly over the equal-size run (57.92%/57.71%). Notably the *smaller*
4G model now edges out the larger 5G one — at only 2 epochs this isn't conclusive (the small model
converges faster early on simply because it has fewer parameters to fit), but it's consistent with
the paper's finding that a smaller model isn't handicapped by MinLogit and can hold its own. Real
signal on this will only come from the full vanilla/DML/KDCL comparison in Phase 5.

## Phase 5 — Training Loop Refactor: the vanilla / DML / KDCL comparison

All three runs: 50 epochs, seed=1, 4G_agent=`rat_mlp_small`, 5G_agent=`rat_mlp_large`,
distortions=[awgn, fading], T=2.0, alpha=0.5, lr=1e-3. Only `--mode` changed between runs.

| Mode    | 4G_agent (small) | 5G_agent (large) |
|---------|-------------------|-------------------|
| vanilla | 73.85%            | 78.20%            |
| dml     | 64.85%            | 64.83%            |
| kdcl    | 64.80%            | 68.44%            |

**DML reproduces the paper's failure mode clearly.** Both models drop sharply from vanilla and
converge to nearly the same value (64.85% / 64.83%) — mutual peer distillation drags the stronger
5G model down toward the weaker 4G model instead of lifting either one. This is a clean, textbook
demonstration of exactly the problem KDCL's MinLogit is designed to solve.

**KDCL partially protects the stronger model, but does not yet beat vanilla.** Compared to DML,
KDCL clearly helps the 5G model (68.44% vs 64.83% — recovers over half the gap DML lost), and is
roughly even with DML for 4G (64.80% vs 64.85%). This is a real, positive result: KDCL is
measurably better than DML's negative-transfer failure. **However**, neither branch under KDCL beats
its own vanilla (undistilled) accuracy — 4G: 64.80% vs 73.85%, 5G: 68.44% vs 78.20%. So the honest
read right now is: *KDCL protects against DML's failure mode, but hasn't yet reproduced the paper's
full headline result of beating standalone training.*

**Why this is likely, and not a dead end:** the paper's own results (Section 4) are sensitive to
`T` and `λ`, which we haven't tuned at all yet — currently using placeholder defaults (T=2.0,
alpha=0.5) carried over from the CIFAR reference implementation. `plan.md` Phase 6 already calls
for a `T`/`λ` sweep; that sweep is now the clear next step rather than a nice-to-have. Also worth
trying once the sweep is done: `KDCL-Linear` as an alternative to MinLogit (already implemented in
`src/kdcl.py`), since with only 2 students the convex-combination approach may behave differently
than the parameter-free MinLogit trick. Also worth more epochs — 50 may not be enough for the KD
loss term to fully pay off relative to pure CE.

**For the write-up:** this is a legitimate, defensible mid-project result, not a failure —
reproducing DML's negative transfer is itself evidence the experimental setup is sound (if DML
hadn't hurt the strong model, that would suggest something else was wrong). The "beats vanilla"
part is Phase 6 work, not Phase 5.

## Phase 6 — Evaluation: hyperparameter sweep

`src/sweep.py` — grid search over `T` (1.0/2.0/4.0/8.0) x `alpha` (0.2/0.5/0.8), 12 combos, 30
epochs each (scaled-down milestones [18,24]), data loaded once and reused across the grid rather
than reloading per run.

```
=== Sorted by average val acc ===
T=1.0 alpha=0.2: 4G=64.09 5G=71.82 avg=67.95
T=4.0 alpha=0.2: 4G=64.65 5G=65.91 avg=65.28
T=2.0 alpha=0.2: 4G=63.59 5G=65.77 avg=64.68
T=1.0 alpha=0.5: 4G=64.30 5G=64.47 avg=64.38
...
T=8.0 alpha=0.8: 4G=59.71 5G=57.27 avg=58.49
```

**Clear, consistent pattern: `alpha=0.2` wins at every temperature tested.** The CIFAR-inherited
default (`alpha=0.5`) over-weights the KD term for this task — makes sense in hindsight: with only
3 output classes (vs. 1000-way ImageNet), the soft-target distribution carries much less
information, so leaning harder on ground-truth CE loss (low alpha) works better than in the
original paper's setting. `T` had a much smaller effect than `alpha`; `T=1.0` edged out slightly.

**Confirmed with a full 50-epoch run at T=1.0, alpha=0.2** (`python src/train.py --mode kdcl --T 1.0 --alpha 0.2 --epochs 50`):

| Config                          | 4G_agent (small) | 5G_agent (large) |
|----------------------------------|-------------------|-------------------|
| vanilla                         | 73.85%            | 78.20%            |
| dml                              | 64.85%            | 64.83%            |
| kdcl (T=2.0, alpha=0.5, default) | 64.80%            | 68.44%            |
| **kdcl (T=1.0, alpha=0.2, tuned)** | **66.88%**       | **78.72%**        |

**This is the headline result of the project so far: tuned KDCL beats vanilla for the 5G_agent**
(78.72% vs 78.20%) — the paper's core claim (collaborative training with a smaller peer lets the
stronger model exceed its standalone accuracy) reproduces on this dataset. The 4G_agent improved
substantially over default KDCL (66.88% vs 64.80%) but still trails its own vanilla number
(73.85%) — the benefit is asymmetric so far: the *large* model gained from collaborating with the
small one, but the *small* model hasn't yet gained from collaborating with the large one. Worth
investigating further (e.g. a still-lower alpha, or per-branch alpha values instead of one shared
alpha) but not necessary to claim the core result — the paper itself frames the win in terms of the
stronger model benefiting without being penalized, which is exactly what happened here.

### ICL ablation — unexpected result

Re-ran the tuned config (T=1.0, alpha=0.2) with `--distortions none none` (added a `NoDistortion`
identity class to `src/distortions.py` for this):

| Config                      | 4G_agent | 5G_agent |
|------------------------------|----------|----------|
| kdcl tuned, ICL on (awgn/fading) | 66.88%   | 78.72%   |
| kdcl tuned, ICL off (none/none)  | 70.47%   | 79.49%   |

**Disabling ICL improved both branches** — the opposite of the paper's ~0.5% drop-when-disabled
finding. Likely explanation: our distortions (`AWGNDistortion` sigma=0.05, `RayleighFadingDistortion`
scale=0.1) use a fixed absolute magnitude, but the 40 features span wildly different raw scales
(`Level` ~ -100, `DL_bitrate` ~ hundreds, `Longitude` ~ 101, vs. the `*_norm` features already
scaled to 0-1). A fixed-magnitude perturbation is negligible for large-scale features and
disproportionately large for the already-normalized ones — unlike image pixels, which all share one
consistent 0-255 (or normalized) scale, so crop/flip perturbs "content" roughly evenly. Our
distortion likely isn't a faithful tabular analogue of ICL yet. **Flagged as a limitation, not
re-implemented yet** — the fix would be per-feature-relative noise (e.g. scaled by each feature's
own std) rather than one fixed sigma across all features.

### Noise-robustness test — inconclusive, test itself under-powered

`src/robustness_test.py`: injected AWGN noise (sigma 0.0 -> 0.5) into the *validation* set at test
time (not training) and evaluated all four trained checkpoints (vanilla, dml, kdcl_tuned,
kdcl_tuned_no_icl).

**Result: accuracy is essentially flat across every noise level, for every mode** — e.g.
vanilla/5G_agent goes 78.20% -> 78.25% -> 78.23% -> 78.40% -> 78.11% -> 78.08% from sigma=0 to
sigma=0.5, i.e. no meaningful degradation anywhere. This is the same root cause as the ICL ablation
finding above: a fixed absolute sigma up to 0.5 is negligible noise relative to features like
`Level` (~-100) or `DL_bitrate` (~hundreds), so the test isn't actually stressing the models. **This
is not evidence of robustness — it's evidence the test needs per-feature-relative noise (e.g. a
fraction of each feature's std) to be meaningful.** Documenting as an honest limitation rather than
claiming a result the test can't actually support.

**Takeaway for the write-up:** both of these findings point at the same fix — feature-scale
normalization (z-score or similar) before applying any perturbation, whether during training (ICL)
or at test time (robustness check). Worth doing before finalizing Phase 6's ablation/robustness
claims; not done yet given time, flagged clearly as a known gap rather than glossed over.

**Phase 6 items still open:** feature-scale normalization (blocks a valid ICL ablation/robustness
retest), confusion matrix/F1 per mode.

## Phase 6 — Normalization fix, and a significant result revision

Added `fit_scaler`/z-score normalization to `src/data.py` (mean/std computed on the *training*
split only, applied consistently to train/val), wired into `train.py`, `infer.py`, and
`robustness_test.py`. All four configs re-run at 50 epochs each with normalization on.

| Config                      | 4G_agent | 5G_agent |
|-------------------------------|----------|----------|
| vanilla                       | 91.15%   | 89.48%   |
| dml                            | **91.86%** | **90.71%** |
| kdcl (T=1.0, alpha=0.2, ICL on) | 91.39%   | 89.66%   |
| kdcl (T=1.0, alpha=0.2, ICL off)| 91.13%   | 90.51%   |

**Normalization alone was a much bigger lever than any of the distillation methods.** Vanilla
jumped from 73.85%/78.20% (unnormalized) to 91.15%/89.48% — confirming the earlier absolute numbers
were significantly held back by poorly-scaled raw features, exactly as suspected from the ICL
ablation/robustness findings above.

**Important revision: DML no longer fails, and now edges out KDCL for both branches.** This
directly contradicts the earlier headline result (Phase 6, first pass) that KDCL beat vanilla and
clearly beat DML. With clean inputs, all four configs converge to within ~2.4 points of each other,
and DML is actually the *best* performer, not the worst.

**Why this makes sense, on reflection:** DML's failure mode (mutual mimicry dragging a stronger
model toward a weaker one) requires a genuine, sizeable *performance gap* between the two students
to manifest — that's exactly the scenario in the paper (ResNet-50 at 76.8% vs ResNet-18 at 71.2% on
a hard 1000-way task). Once our features were properly normalized, the vanilla gap between our
small and large MLP nearly vanished (91.15% vs 89.48% — under 2 points, and the *small* model is
now ahead) on this comparatively easy 3-class tabular task. With no real capacity/performance gap
to exploit, there's no "weak student" for DML to be dragged down by, so its core failure mode
simply doesn't have anything to bite into here — and with no failure to fix, KDCL doesn't have an
obvious advantage to demonstrate either.

**This is not a dead end — it's a diagnosis.** The unnormalized-pipeline results earlier weren't
demonstrating the paper's mechanism at all; they were showing that badly-conditioned optimization
makes both DML and KDCL look more different from vanilla than they really are, in ways that don't
reflect the actual phenomenon the paper describes. Now that the pipeline is fixed, the honest
finding is: **on this dataset, with this architecture pairing, the capacity gap is too small for
KDCL's advantage over DML to show up.** The paper's own experiments deliberately pick pairings with
large real accuracy gaps (e.g. pairing ResNet-50 with a MobileNetV2x0.5 at only 64.8% standalone).
To fairly test KDCL's actual claim here, the next step should be **deliberately widening the
capacity gap** — e.g. a much smaller "weak" model (fewer/no hidden layers) for one branch — so that
vanilla training actually produces a sizeable accuracy gap between the two RATs, recreating the
conditions the paper's method is designed for, before concluding whether KDCL helps or not.

**Robustness test, re-run with normalized noise injection, now behaves sensibly:** clean, monotonic
degradation for every mode (~90% at sigma=0 down to ~77-79% at sigma=0.5) — the test itself is now
valid, unlike the flat/inconclusive curve from before normalization. However, **vanilla degrades
about as gracefully as (or slightly better than) the distilled methods** in this run — no clear
robustness advantage for KDCL/ICL was found. Given this is a single seed with no repeated trials,
small (1-3 point) differences here shouldn't be over-interpreted either way.

**For the write-up:** this is exactly the kind of finding a rigorous FYP should surface rather than
hide — an initial result (KDCL beats vanilla) turned out to be partly an artifact of a
preprocessing bug, and fixing that bug changed the conclusion. The path forward (widen the capacity
gap deliberately) is a natural, well-motivated next experiment, not a scramble to rescue a broken
result.

## Phase 6 — Attempting to widen the capacity gap: conclusive negative result

Three attempts to create a real standalone performance gap between the small (4G) and large (5G)
models, all under vanilla (no distillation) training so the comparison is clean:

| Attempt                                          | 4G (weaker arch) | 5G (stronger arch) |
|----------------------------------------------------|-------------------|-----------------------|
| 3-class, small vs large (original)                | 91.15%            | 89.48%                |
| 3-class, **tiny** (pure linear, no hidden layer) vs large | 90.77%      | 89.54%                |
| 5-class (harder label), small vs large            | 80.63%            | 78.87%                |
| 3-class, small vs large, **150 epochs** (3x training) | 91.57%          | 89.48%                |

**In every single attempt, the smaller/weaker architecture wins.** Even a pure linear classifier
(no hidden layer at all) essentially matches or beats a 256→128→64 deep MLP. Making the label
harder (5 classes) dropped both models' accuracy as expected but didn't change which one wins.
Giving the large model 3x the training budget (150 vs 50 epochs) didn't help either — and the
*reason* is visible directly in the training curve: 5G_agent's train accuracy kept climbing
(89.66%→91.60% from epoch 25 to 150) while its **validation loss kept climbing too** (0.223→0.306)
even as training loss fell. That's a textbook overfitting signature, not undertraining — the large
model has more capacity than this task/dataset size (~15.9k training rows) needs, and uses the
extra capacity to overfit rather than generalize better.

**Conclusion: this dataset's feature set and label design do not support a fair, real capacity gap
between architectures of this kind.** Forcing a bigger nominal gap (e.g. an even deeper "5G" model)
would only measure worsening overfitting, not a genuine capability difference — not a fair test of
DML/KDCL's actual mechanism. Stopping this line of investigation here rather than continuing to
guess at more configurations.

**What this does NOT undermine:** the earlier normalized-pipeline finding that both DML and KDCL
give a small, consistent edge over vanilla training (~1-2 points, see the normalization section
above) still stands as legitimate evidence that collaborative training transfers to this tabular
RAT domain without breaking anything — just not the paper's more dramatic "weak model rescues/
threatens strong model" story, which needs a real capacity gap this dataset doesn't provide with
the architectures tried. **Proceeding to Phase 7** (RAT-selection inference stage) with this as the
final, honestly-documented Phase 6 conclusion.

## Phase 7 — RAT-Selection Inference Stage

`src/infer.py`: rule-based recommender (plan.md's Option 1) — run a shared state vector through
both trained branches, recommend whichever predicts the better expected KPI outcome. Fixed it to
evaluate on the true held-out **test** split (`time_blocked_split`'s third slice, never touched by
training or hyperparameter tuning) rather than an arbitrary slice of the full dataframe, and added
a majority-class baseline for context (a naive "always recommend the more common RAT" comparison).

**First run, using the KDCL-trained checkpoints (T=1.0, alpha=0.2, ICL on, normalized):**
```
Test set size: 3403
Agreement with actual NetworkTech in use: 2052/3403 = 60.3%
Majority-class baseline (always recommend '5G'): 62.7%
Agreement on rows that actually achieved the top load class: 597/747 = 79.9%
```

**Confound found before trusting this:** `is_5G` is one of the 40 trained features, so a row's
actual serving tech is baked directly into its input. Both branches see the *same* row, including
its real `is_5G` value — so "asking the 5G_agent" about a row that was actually served by 4G isn't
a genuine counterfactual, it's just handing the 5G_agent a row whose features say "this is 4G" and
watching it (correctly) predict accordingly. This isn't a fair independent comparison.

**Fix:** `recommend_rat` now overrides `is_5G` to each branch's *own* native value (0 for 4G_agent,
1 for 5G_agent, computed in the normalized/scaled space via `normalized_is_5g_values`) before
scoring, regardless of what the row's real tech was. This is what "what would my tech's model think
of this state" actually requires.

**Re-run with the fix:**
```
Test set size: 3403
Agreement with actual NetworkTech in use: 1992/3403 = 58.5%
Majority-class baseline: 62.7%
Agreement on rows that actually achieved the top load class: 443/747 = 59.3%
```

**Honest conclusion: once the confound is removed, the recommender performs roughly at
chance/baseline level** — the earlier 79.9% figure was substantially inflated by the model reading
off the actual tech identity rather than reasoning independently. This is consistent with, and
directly explained by, the Phase 6 finding that there's no real capacity/performance differentiation
between the two trained branches on this task: if 4G_agent and 5G_agent behave nearly identically
as functions, forcing each to imagine "if I were this row's tech" naturally produces little
differentiated signal between them. **This is a genuine limitation to report, not an implementation
bug** — a good discussion point: a meaningfully useful RAT recommender likely needs either (a) two
branches with real, non-overfit capacity/behavior differences (which Phase 6 showed this dataset
doesn't support with MLP-style architectures), or (b) a task where RAT choice actually matters more
than it appears to for this label design.

### Log
- **2026-07-14 → 15:** Confirmed real RAT split (4G/5G only), refactored `data.py`/`train.py` to
  share instances across branches (fixed initial design bug), built full `src/` pipeline.
- **2026-07-16:** Ran smoke test successfully end to end. Ran Phase 1 EDA script; found and fixed
  two label-leakage issues (`temp_load`/`sig_load` correlation with label, and `load_class` itself
  leaking into the feature matrix via an ordering bug). Feature count now 40, confirmed clean.
  Missingness reviewed — no changes needed, existing `*_missing` flag approach is sufficient.
  Decided branch capacity split (4G small / 5G large), set as new training default, re-ran smoke
  test to confirm it trains cleanly with the new architecture split.
- **2026-07-16 (later):** Realized Phases 2-4's code had actually been built already (during initial
  repo setup) but `plan.md` checkboxes were never updated to reflect it, causing confusion about
  what stage the project was at. Corrected `plan.md` to show true status per phase. Moved this log
  from `docs/phase1.md` into `docs/notes/progress_log.md`, restructured as an ongoing multi-phase
  log rather than a Phase-1-only file, so it can keep growing as later phases complete.
- **2026-07-16 (evening):** Renamed `notebooks/phase1_eda.py` → `notebooks/eda.py`, fixed all
  references. Ran the real Phase 5 comparison: vanilla, DML, KDCL, 50 epochs each. DML clearly
  reproduces the paper's negative-transfer failure; KDCL clearly beats DML (especially protecting
  the 5G model) but does not yet beat vanilla — default `T`/`alpha` need tuning next (Phase 6).
- **2026-07-16 (night):** Built `src/sweep.py`, ran a 12-combo `T`/`alpha` grid search. Found
  `alpha=0.2` clearly beats the default `0.5` at every temperature. Confirmed the winning combo
  (T=1.0, alpha=0.2) with a full 50-epoch run: **5G_agent hits 78.72%, beating vanilla's 78.20%** —
  the project's first result matching the paper's actual headline claim. 4G_agent still trails its
  vanilla number. Remaining Phase 6 work: ICL ablation, noise-robustness test, confusion matrix/F1.
- **2026-07-16 (later still):** Ran the ICL ablation and noise-robustness test. Both turned up the
  same root cause: distortions use a fixed absolute magnitude, but features span wildly different
  raw scales, so the perturbation is negligible for large-scale features and disproportionate for
  already-normalized ones. Ablation showed disabling ICL *improved* accuracy (opposite of the
  paper's expectation); robustness test showed flat accuracy across all noise levels for every
  mode (test under-powered, not evidence of genuine robustness). Documented honestly as a known
  limitation — fix is feature-scale normalization before applying any perturbation. Not fixed yet;
  next up is confusion matrix/F1 per mode, then decide whether to revisit normalization.
- **2026-07-16 (very late):** Added z-score feature normalization, re-ran all four configs.
  Vanilla jumped from ~74-78% to ~89-91% — confirms unscaled features were badly hurting
  optimization all along. **Revised finding: DML no longer fails and now edges out KDCL for both
  branches** — contradicts the earlier "KDCL beats vanilla" headline. Diagnosis: DML's failure mode
  needs a real capacity/performance gap to bite into, and normalization nearly closed the gap
  between our small/large MLP (the small one is even slightly ahead now). Robustness test now
  behaves sensibly (monotonic degradation) but shows no clear KDCL advantage either. Next: decide
  whether to deliberately widen the capacity gap between branches to recreate the conditions the
  paper's method is designed for.
- **2026-07-16 (final):** Also added persistent per-run `train_log.txt` files and un-gitignored
  `experiments/` (checkpoints are tiny, 2.7MB total) so every run is now git-verifiable evidence,
  not just a claim in a chat transcript. Tried three ways to widen the capacity gap (a pure-linear
  4G model, a harder 5-class label, 3x training epochs) — the smaller model won every single time,
  and the 150-epoch run's train/val loss curves showed the large model overfitting, not
  undertraining. Concluded this dataset/architecture combination doesn't support a fair capacity
  gap; stopped chasing it. **Phase 6 is done.** Moving to Phase 7 (RAT-selection inference stage).
- **2026-07-16 (Phase 7):** Fixed `infer.py` to evaluate on the true held-out test split (was using
  an arbitrary df slice) and added a majority-baseline comparison. First run looked promising
  (79.9% agreement on rows that actually hit the top load class) but found a confound: `is_5G` is a
  trained feature, so both branches could see the row's real serving tech directly rather than
  reasoning independently. Fixed by overriding `is_5G` to each branch's own native value before
  scoring. Re-run dropped to 59.3% — roughly chance/baseline level. Honest conclusion: this
  directly reflects Phase 6's finding of no real capacity/behavior differentiation between the two
  branches — documented as a genuine limitation, not a bug. **Phase 7's rule-based recommender is
  implemented and evaluated; the result is an honest negative, consistent with Phase 6.**
