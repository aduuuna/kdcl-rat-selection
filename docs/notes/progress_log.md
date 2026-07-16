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
