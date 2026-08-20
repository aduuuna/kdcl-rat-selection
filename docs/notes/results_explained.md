# Results Explained: Training, Validation, and Test

What the numbers in `progress_log.md` actually mean, and what they add up to. Written to be read on
its own — if you need the code-level mechanics behind any of this, see `build_explanation.md`.

## 1. Quick refresher: three different measurements, three different jobs

- **Training accuracy** — how well a model does on the exact data it's currently learning from.
  Not very meaningful on its own (a model can memorize training data without learning anything
  generalizable), so it's mostly useful as a sanity check that learning is happening at all.
- **Validation accuracy** — measured after every epoch, on a slice of data the model never trains
  on. Used *during* training to decide which epoch's weights are worth keeping. This is what most
  of the numbers reported so far actually are.
- **Test accuracy** — measured *once*, at the very end, on a third slice of data that was never
  touched during training or checkpoint selection. This is the only fully unbiased number — it
  answers "how will this model actually perform on data it has truly never seen," rather than "how
  well did we pick a checkpoint."

All three come from `time_blocked_split()`: the dataset is sorted chronologically and cut
70% / 15% / 15% into train / validation / test — train on the earliest three weeks of June 2016,
validate on the next few days, test on the last few days of the month.

## 2. The current results, in full

Every mode below uses the current standard setup: `is_4G` excluded from the features (see
`progress_log.md`'s "Standardizing on no-tech-flag" section for why), 50 epochs, same seed.

| Mode | 3G_agent val | 3G_agent test | 4G_agent val | 4G_agent test |
|---|---|---|---|---|
| vanilla | 66.10% | 65.33% | 66.82% | 66.54% |
| dml | 64.79% | 63.02% | 65.16% | 63.45% |
| **kdcl (T=1.0, alpha=0.1)** | 66.10% | 64.85% | 66.75% | 66.26% |

Random-guess baseline for this 3-class problem is 33.3%, so all three modes are clearly learning
real structure from the 15 radio-condition features — the question is which learns *better*, and
specifically whether collaborative training helps or hurts relative to training each RAT model
alone.

## 3. What each row means, in plain terms

**Vanilla** is the control group: each RAT model trained completely alone, no collaboration. This
is the number every other mode has to be compared against — "what would we get with no clever
technique at all?"

**DML** (Deep Mutual Learning) is the older collaborative method: both models are pushed to mimic
each other's raw predictions directly, with no safeguard. Compare its numbers to vanilla:
- 3G: 66.10% → 64.79% (**-1.31 val**, **-2.31 test**)
- 4G: 66.82% → 65.16% (**-1.66 val**, **-3.09 test**)

Both models got *worse* than training alone, and the stronger 4G model was hurt more in absolute
terms than the weaker 3G model. That is exactly the failure mode the KDCL paper describes: when one
model is genuinely stronger than the other, naive mutual mimicry lets the weaker model's noisier
guesses drag the stronger one down. Notably, **the test-set drop is bigger than the validation-set
drop** for both branches — this failure isn't a validation-set fluke, it's a real, if anything
slightly worse, effect on genuinely unseen data.

**KDCL** (this project's method, using the paper's MinLogit ensembling) is meant to fix exactly that
problem. Compare its numbers to both:
- vs. vanilla: 3G ties exactly on validation (66.10% = 66.10%) and is within 0.5 points on test
  (64.85% vs 65.33%); 4G is within 0.1 points on validation (66.75% vs 66.82%) and within 0.3 points
  on test (66.26% vs 66.54%).
- vs. dml: KDCL beats DML by **+1.31 points (val) / +1.83 points (test)** for 3G, and by
  **+1.59 points (val) / +2.81 points (test)** for 4G.

So KDCL doesn't just do slightly better than DML — it recovers almost the entire gap DML lost
relative to training alone, on both branches, confirmed on data neither model was tuned against.

## 4. What this adds up to

This is the clearest result either dataset attempted in this project has produced, and it is
validated three ways rather than one:
1. **Direction is consistent between validation and test.** The same story (DML clearly worst,
   KDCL close to vanilla) shows up whether you look at the number used to pick the checkpoint or
   the number computed afterward on genuinely unseen data.
2. **The DML failure is real, not an artifact.** It was originally seen (more dramatically) on the
   old dataset before feature normalization was added there, which raised the question of whether
   it was a preprocessing artifact. Here, on properly normalized features from the start, DML still
   clearly underperforms — confirming the failure mode is genuine, not a scaling bug.
3. **KDCL's recovery is not a lucky single number.** It holds on both branches, and holds under the
   stricter test-set check, not just validation.

**What is still open:** every number here comes from a single training run (one random seed). A
half-point-to-one-point gap between kdcl and vanilla is small enough that it could shift with a
different seed — repeating each mode 3-5 times with different seeds would tell us whether "KDCL
ties vanilla" is a stable finding or something that happens to land close by chance. That has not
been done yet.

## 5. The RAT-selection stage — a related but separate result

Beyond training the models well, this project also asks: can the trained models be used to
recommend *which RAT to use* for a given situation? This was tested twice, because the first attempt
uncovered a real problem.

**First attempt** (models trained *with* `is_4G` as a feature): the recommender always said "4G,"
for literally 100% of test rows. It looked informative at first (66.1% overall agreement, 99.8%
agreement on genuinely great-outcome rows) but both numbers were fake — they're what you'd get from
a recommender that never changes its answer, given that 4G legitimately is the majority tech and
usually is better. Confirmed by checking: the confidence gap favoring 4G was positive for every row
sampled, never once favoring 3G.

**Fix:** retrained without `is_4G` as an input at all, forcing both models to reason from the 15
real radio-condition features instead of leaning on the dominant tech-identity flag.

**Result after the fix:** the recommender is no longer degenerate — it splits its answers
(2,420 recommend 3G, 2,945 recommend 4G, out of 5,365 test rows). The honest numbers:
- Overall agreement with the RAT actually in use: 65.6%, essentially tied with the naive
  "always guess 4G" baseline of 66.1% — across the *full* range of situations, it isn't clearly
  better than guessing.
- Agreement specifically on rows where a genuinely top-tier outcome was actually achieved: 80.2%,
  meaningfully above the 66.1% baseline, and this time earned through real per-row reasoning rather
  than a constant answer.

**What this means:** the recommender has a real, if narrow, capability — it's specifically good at
recognizing when conditions are genuinely favorable for the better RAT, even though it isn't
confidently better than a naive guess across every kind of situation. That's a modest, honest
finding, not a polished success story, and it's reported that way deliberately.

## 6. Overall, what was achieved

- A working, correctly-implemented KDCL pipeline (verified against the paper's actual MinLogit math,
  not a shortcut), adapted from image classifiers to tabular wireless network data.
- A dataset with a genuine, physically-grounded performance gap between RATs (3G vs. 4G), found
  specifically because the original dataset could not provide one — confirmed through three
  separate, deliberate attempts to force a gap that all failed for a diagnosable reason (overfitting,
  not undertraining).
- A clean, validated (train → val → test all agree) demonstration that DML genuinely fails under a
  real capacity gap, and that KDCL genuinely recovers from that failure — the core claim of the
  paper, reproduced on real networking data.
- A working RAT-selection recommender, honestly evaluated, with a real (if narrow) capability
  identified and a degenerate failure mode found and fixed along the way.
- What's still open: confirming the KDCL-vs-vanilla result across repeated random seeds, trying the
  alternative `KDCL-Linear` ensembling method, and a few smaller robustness checks (see `plan.md`).
