# Understanding Your Project: A Guided Walkthrough

You're presenting tomorrow, so this is written to be read start to finish, in plain language, with
no assumed context. It covers: the big picture, what every file does, what we actually achieved, and
a slow, careful walkthrough of the validation stage since that's the part you said feels least clear.

---

## Part 1: The Big Picture — what are we actually doing?

Imagine two students studying for the same exam: one is decent, one is stronger. The old way to make
the weaker one better is to have the strong one become a "teacher" first — fully finish learning,
then explain everything to the weaker student. That's slow (the teacher must finish first) and
one-directional (the teacher never gets anything back).

**KDCL's idea**: instead, both students study *together*, at the same time. After every practice
question, they compare answers and build a combined "best answer" from both of them, then both learn
from that combined answer as well as the real answer key. Done carefully (KDCL's "MinLogit" trick),
this combined answer is never worse than either student's own answer alone — so even the weaker
student can never drag the stronger one down, and sometimes the weaker student's different
perspective actually helps the stronger one.

**Your project's question**: does this same trick work if the "students" aren't image-recognition
models, but two small neural networks — one representing 4G, one representing 5G — learning to
predict network conditions (like whether load will be Low/Medium/High) from real drive-test data?

That's it. That's the whole project in one paragraph. Everything else is either (a) building the
machinery to test this, or (b) the honest results of testing it.

---

## Part 2: The codebase, file by file

Think of `src/` as a pipeline — data flows through these files roughly in this order:

### `src/data.py` — loads and prepares the data
Reads the Excel file (22,690 rows of real drive-test measurements), figures out which columns are
safe to use as inputs (excludes ID columns, timestamps, and — importantly — a couple of columns that
turned out to be near-duplicates of the answer we're trying to predict, more on that in Part 3),
builds the Low/Medium/High label, and standardizes every number onto the same scale (more on why
that mattered later). It also defines `MultiViewDataset` — the class that makes sure **both the 4G
model and the 5G model see the exact same row of data**, each with its own bit of simulated noise
added, rather than each model only seeing "its own" rows. This mirrors exactly how the two students
in the analogy above both see the same practice question.

### `src/distortions.py` — the simulated noise
Small classes that add fake radio-channel imperfections (random noise, signal fading, dropped
readings) to a row of data before a model sees it. This is the tabular-data equivalent of the
original KDCL paper randomly cropping or flipping an image before a model sees it — the idea is to
make each model see a slightly different, slightly harder version of the same situation, so they
don't just memorize identical inputs.

### `src/models.py` — the actual neural networks
Small, simple networks (a few linear layers) that take a row of ~40 numbers and output a prediction
across the 3 load classes. We have a few sizes available (`rat_mlp_small`, `rat_mlp`,
`rat_mlp_large`, `rat_mlp_tiny`) so we can assign a smaller network to one RAT and a bigger one to
the other, matching the "one strong student, one weaker student" setup KDCL is designed around.

### `src/kdcl.py` — the actual "combine both students' answers" math
This is the heart of the technique. Three ways to combine the two models' predictions into one
shared "best answer":
- **naive**: just trust whichever model got this specific example most correct.
- **minlogit**: the paper's real trick — mathematically guaranteed to never be worse than either
  model alone. This is the one we use by default.
- **linear**: a weighted blend of both models' answers, where the weights are learned.

### `src/train.py` — runs the actual training
This is what you actually run. It has three modes you can switch between with `--mode`:
- `vanilla`: each model trains completely alone, no collaboration. This is our baseline — "what if
  we didn't use KDCL at all?"
- `dml`: an older collaboration method (Deep Mutual Learning) that just has both models mimic each
  other directly, with no clever combining step. This is what KDCL is supposed to improve on.
- `kdcl`: our actual method — both models train together, supervised by the combined "best answer"
  from `kdcl.py`.

### `src/sweep.py` — searches for the best settings
KDCL has two dials to tune: `T` (temperature — how "soft" the combined answer is) and `alpha` (how
much weight to give the combined answer vs. the real answer key). This script tries 12 combinations
automatically and reports which worked best, so we don't have to guess by hand.

### `src/robustness_test.py` — checks how models handle noisy input
After training, this feeds increasingly noisy versions of the test data into each trained model and
checks how much accuracy drops. The idea (from the paper) is that KDCL-trained models should degrade
more gracefully under noise than models trained alone.

### `src/infer.py` — the "which RAT should we pick?" stage
This is the part you said doesn't feel clear yet — it gets a full plain-language walkthrough in
Part 4 below.

### `src/utils.py`
Small shared helper functions (nothing conceptually interesting — just bookkeeping for tracking
accuracy numbers during training).

### Everything outside `src/`
- `data/raw/urban_dataset_processed.xlsx` — the real dataset.
- `docs/paper/` — the original KDCL research paper.
- `docs/notes/progress_log.md` — a day-by-day log of everything we tried, including the mistakes and
  how we caught them. Good for showing your working process.
- `docs/notes/writeup.md` — the same story, but reorganized into thesis-chapter form (Background,
  Methodology, Results, Discussion) instead of chronological order.
- `docs/notes/explanation.md` — this file.
- `experiments/` — every training run we've done is saved here: the trained model weights, plus a
  text log of exactly what happened epoch by epoch. This is your proof that these experiments
  actually ran, not just claims.
- `plan.md` — the phased project roadmap, with checkboxes showing what's done.
- `notebooks/eda.py` — the script we used to explore the data and check for problems before trusting
  any results (explained more in Part 3).
- `reference/` — the original image-based KDCL code we adapted from. Kept for comparison, not used
  directly.

---

## Part 3: What did we actually achieve?

Told honestly, in order:

**1. We built a working pipeline** that takes real 4G/5G drive-test data, trains two small neural
networks collaboratively using the paper's actual MinLogit method (not a shortcut — we specifically
checked the reference code we started from was *not* doing real MinLogit, and fixed that), and can
compare that against training alone (vanilla) and against the older DML method.

**2. We found and fixed two real bugs before trusting any results.** First, two columns
(`temp_load`, `sig_load`) turned out to be near-duplicates of the answer we were predicting, so we
removed them. Second — more seriously — we found the actual answer label had accidentally leaked
into the model's input features due to an ordering mistake in the code, meaning early accuracy
numbers were the model partly "cheating" by seeing its own answer. We caught this, fixed it, and
re-ran everything. This kind of catch-and-fix is exactly the sort of thing worth mentioning in a
presentation — it shows the results are trustworthy, not just optimistic.

**3. Feature scaling turned out to matter enormously.** Our 40 input columns have wildly different
number ranges (one column might range -100 to 0, another 0 to 1). Neural networks train much better
when all inputs are on a similar scale. Once we fixed this (standardizing every column), accuracy
jumped from around 74-78% to around 89-91% — a bigger improvement than any of the fancy distillation
methods gave us. **Lesson for your presentation: basic data preprocessing mattered more than the
sophisticated technique.** That's a genuinely interesting finding, not a footnote.

**4. Once we fixed the scaling, our comparison of methods changed.** Before the fix, DML (the older
method) clearly failed — badly, exactly as the paper predicts happens when one model is much weaker
than the other. KDCL clearly beat it. After the fix, DML stopped failing entirely and actually
slightly beat KDCL. We investigated why: **DML only fails when there's a real skill gap between the
two models, and once our data was properly scaled, our "weak" and "strong" models turned out to
perform almost identically anyway** — so there was no real gap for DML to fail on, and nothing
dramatic for KDCL to fix.

**5. We tried three different ways to force a real skill gap to exist**, so we could fairly test
whether KDCL actually helps when it's supposed to: making one model much simpler (just a single
layer, no hidden layers at all), making the prediction task harder (5 categories instead of 3), and
giving the bigger model 3x more training time. **In every single attempt, the smaller model still
matched or beat the bigger one.** We even confirmed why in the last attempt: the bigger model's
performance on new data was getting *worse* the more we trained it, even though it was getting
better on data it had already seen — a classic sign of a model that's too complex for the amount of
data available, not a model that needs more time.

**6. Conclusion: KDCL "works" here in a modest, real sense** (both KDCL and DML give a small,
consistent boost over training alone), **but this particular dataset didn't let us prove the paper's
more dramatic claim** — that a weak model can meaningfully help, without ever hurting, a genuinely
stronger one — because we couldn't create a genuine strength gap between our two RAT models no
matter how we tried. This is a legitimate, honestly-earned scientific conclusion, not a failure.
Presenting "here's what we tried to prove it further, and here's why it didn't work, and here's why
that's still a real finding" is a strong thing to say in a viva or presentation.

---

## Part 4: The validation stage, explained slowly

This is Phase 7 of the project — `src/infer.py` — and it's a genuinely separate question from
everything above. Everything in Part 3 was about *training* the two models well. This part is about
*using* the trained models to actually make a decision: **given a network situation right now, which
RAT should be recommended?**

### Step by step, what the code actually does

1. **Load the two trained models** (the 4G one and the 5G one) from their saved checkpoint files.
2. **Take a row from the test set** — a network state we've never used for training or tuning, so
   this is a fair test. (This matters: earlier versions of this code accidentally tested on data
   that wasn't properly held out — we caught and fixed that too.)
3. **Feed that same row to both models.** Each model outputs a probability for "Low load," "Medium
   load," "High load."
4. **Turn that into a single "how good is this" number** for each model — roughly, "on average,
   which load class does this model expect?" A model that's confident about "High load" scores
   higher than one that's unsure or expects "Low load."
5. **Recommend whichever model scored higher** — i.e., recommend the RAT whose model predicts the
   better outcome for this situation.
6. **Check whether that recommendation matches reality**: for that same row, we know which RAT was
   *actually* in use when the measurement was taken. Does our recommendation agree?

### The subtle problem we found (and why it matters)

Here's the part that took real thought. One of the 40 input columns is literally a yes/no flag for
"was this row actually measured on 5G?" That column is genuinely useful for *training* — it helps
each model learn what's normal for its own RAT. But it becomes a problem at the *recommendation*
step: if we hand the 5G model a row that says "this was NOT measured on 5G," the model can just
read that flag and go "oh, this isn't my situation," without doing any real reasoning about the
actual signal quality. It's like asking someone "would you enjoy this restaurant?" while the menu
literally says "you have already decided you dislike this restaurant" — you're not testing their
genuine judgment anymore, you're just watching them read a label.

**The fix**: when we ask the 4G model to score a row, we force that flag to say "this is a 4G
situation" — and when we ask the 5G model, we force it to say "this is a 5G situation" — regardless
of what actually happened. That way both models are being asked the *same* honest question:
"pretend this state happened on your network — how good would it be?"

### What we found, and why it's still a good result to present

Before the fix, our recommendation agreed with what actually happened about 60% of the time, and
looked especially good (80%) specifically on rows where things went really well. After the fix, that
dropped to around 58-59% — close to what you'd get by just always guessing the more common RAT
without looking at the data at all (about 63%).

**This isn't a coding failure — it's an honest, connected finding.** It directly follows from Part
3's conclusion: if the two trained models don't behave very differently from each other in the first
place (which we already showed), then asking them to "disagree" about which RAT is better naturally
doesn't produce much useful signal either. A recommender built on two nearly-identical opinions can't
meaningfully recommend one over the other. **Once you've explained Part 3's finding, this result
isn't surprising — it's the expected consequence, and finding it (and catching the confound that
would have hidden it) is itself good, careful work.**

---

## Quick reference for your presentation

If you only remember five sentences, remember these:

1. We adapted a real image-classification technique (KDCL) to work on tabular wireless network data
   by treating each RAT as a "student" that learns collaboratively with the other, using the paper's
   actual MinLogit math, not a shortcut.
2. We found and fixed two real bugs (label leakage, and a feature-scaling issue) before trusting any
   result — feature scaling alone was the single biggest factor in model accuracy.
3. Both KDCL and the older DML method give a small, real improvement over training each RAT model
   alone — but we could not create a genuine strength gap between our two RAT models despite three
   separate deliberate attempts, so we couldn't test the paper's more dramatic claim about protecting
   a strong model from a much weaker one.
4. We built and evaluated a RAT-recommendation stage; after fixing a subtle data leak in how we
   tested it, its recommendation quality is close to a naive baseline — an honest result that follows
   directly from finding #3, not a separate failure.
5. Every experiment is saved on disk with full logs and model weights, committed to the repo, so it's
   verifiable, not just claimed.
