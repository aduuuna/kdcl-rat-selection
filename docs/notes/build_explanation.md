# Build Explanation: How This Project Actually Works

Written to be read with the code open next to it. Every section names the exact file and
function/class it's describing, and quotes the real code so you can cross-check line by line —
this is for your defense, so nothing here should require trusting a paraphrase.

## Map: concept → file → function/class

| Concept | File | Function / class |
|---|---|---|
| Load the raw CSV, filter to 3G/4G, build features | `src/netztest_data.py` | `load_raw()` |
| Decide which columns are usable inputs | `src/netztest_data.py` | `numeric_feature_columns()` |
| Turn `download_kbit` into Low/Medium/High | `src/data.py` (reused) | `make_load_label()` |
| Standardize features to mean-0/std-1 | `src/data.py` (reused) | `fit_scaler()` |
| Split into train/val/test by time, not randomly | `src/data.py` (reused) | `time_blocked_split()` |
| Give both branches the same row, each corrupted differently | `src/data.py` (reused) | `MultiViewDataset` |
| The neural networks themselves | `src/models.py` | `RATMLP`, `model_dict`, `build_model()` |
| The simulated channel noise | `src/distortions.py` | `AWGNDistortion`, `RayleighFadingDistortion` |
| The MinLogit / DML / vanilla math | `src/kdcl.py`, `src/train_netztest.py` | `minlogit()`, `teacher_logit_for_mode()`, `compute_losses()` |
| The actual training loop | `src/train_netztest.py` | `train_one_epoch()`, `evaluate()`, `main()` |

---

## 1. Data: `src/netztest_data.py`

`load_raw()` (lines 47-54):
```python
def load_raw(path: str = RAW_CSV, rats: tuple = RATS) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["cat_technology"].isin(rats)].reset_index(drop=True)      # keep only 3G/4G rows
    df = df[df["implausible"] == False].reset_index(drop=True)           # drop 19 flagged-bad rows
    df["is_" + rats[1]] = (df["cat_technology"] == rats[1]).astype(np.float32)  # → "is_4G" column
    for col in SPARSE_COLS:
        df[col + "_missing"] = df[col].isna().astype(np.float32)         # 6 new missing-flag columns
    return df
```
This is the entire raw-loading step. `RATS = ("3G", "4G")` (line 16) is why the flag is named
`is_4G` — it's built from `rats[1]`, the second element.

`numeric_feature_columns()` (lines 57-60) decides which columns the models are actually allowed to
see:
```python
def numeric_feature_columns(df: pd.DataFrame) -> list:
    excluded = NON_FEATURE_COLS | LEAKY_COLS | {LOAD_COL, LABEL_COL}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in excluded]
```
`NON_FEATURE_COLS` (lines 20-28) is IDs/metadata/empty columns. `LEAKY_COLS` (line 40) is
`{"ndt_download_kbit", "ndt_upload_kbit", "upload_kbit"}` — excluded specifically because they
correlate 0.77-0.94 with the label, i.e. near-duplicates of the answer. `LOAD_COL` = `"download_kbit"`
(the raw label source) and `LABEL_COL` = `"load_class"` (the binned label itself) are always
excluded too — a model must never see either the exact answer or the discretized answer as an input.
**Run `python src/netztest_data.py` yourself right now and you'll get the printed list of exactly
16 surviving column names** — that's the ground truth, not this document.

`make_load_label()` lives in `src/data.py` (line 33-34), reused as-is:
```python
def make_load_label(df, load_col=LOAD_COL, n_classes=N_LOAD_CLASSES) -> pd.Series:
    return pd.qcut(df[load_col], q=n_classes, labels=False, duplicates="drop")
```
`pd.qcut` sorts every row by `download_kbit` and cuts it into `n_classes` (3) groups of *equal size*
— that's why the overall class balance is almost exactly 33/33/33 regardless of dataset. Labels are
`0, 1, 2` for Low/Medium/High. **Important:** because this default `load_col=LOAD_COL` refers to
`data.py`'s own constant (`"derived_load"`, the *old* dataset's column), `train_netztest.py` must
call it with `load_col=LOAD_COL` explicitly (netztest's own constant, `"download_kbit"`) — this is
literally the bug I hit and fixed earlier; if you ever see `KeyError: 'derived_load'` again, this is
why.

`fit_scaler()` (`src/data.py` lines 43-51):
```python
def fit_scaler(df, feature_cols):
    values = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std
```
Computed **only on the training split** (see `build_loaders()` below) — this is deliberate: if you
computed mean/std over train+val+test together, information about the val/test rows' distribution
would leak into how every row gets scaled, which counts as a form of data leakage.

`time_blocked_split()` (`src/data.py` lines 59-69) sorts by a time column and slices sequentially
(70% train / 15% val / 15% test) — **not** a random shuffle, because rows close together in time are
correlated (similar network conditions), and a random split would let near-duplicate rows leak
across train/val/test.

`MultiViewDataset` (`src/data.py` lines 72-90) is the class that actually produces what a training
step sees:
```python
def __getitem__(self, idx):
    base = self.features[idx]
    views = np.stack([d(base.copy()) for d in self.distortions])   # one distorted copy per branch
    return torch.from_numpy(views), int(self.labels[idx])
```
For row `idx`, it makes one distorted copy of that same 16-number vector *per distortion function
passed in* (two, for our two branches) and returns them stacked together with **one shared label**.
This is the technical mechanism behind "both branches see the same row."

---

## 2. Models: `src/models.py`

```python
class RATMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(128, 64), dropout=0.2):
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
```
`rat_mlp_small` = `hidden=(64, 32)` → `16 → 64 → 32 → 3`. `rat_mlp_large` = `hidden=(256, 128, 64)`
→ `16 → 256 → 128 → 64 → 3`. Every hidden layer is `Linear → BatchNorm1d → ReLU → Dropout(0.2)`;
the final `Linear` layer has no activation — its raw output *is* the logits (3 real numbers, not
probabilities yet). `model_dict` (lines 27-32) is just a name-to-constructor lookup table;
`build_model(name, in_dim, n_classes)` looks a name up and calls it.

---

## 3. Distortions: `src/distortions.py`

```python
class AWGNDistortion:
    def __call__(self, x):
        noise = self.rng.normal(0.0, self.sigma, size=x.shape).astype(np.float32)
        return x + noise                                    # additive Gaussian noise, sigma=0.05

class RayleighFadingDistortion:
    def __call__(self, x):
        fade = self.rng.rayleigh(scale=self.scale, size=x.shape).astype(np.float32)
        return x * (1.0 - fade)                              # multiplicative fading, scale=0.1
```
These run on the **already-standardized** feature vector (mean-0/std-1 space), so `sigma=0.05` means
"5% of one standard deviation of noise," consistently across all 16 features. `3G_agent` gets AWGN,
`4G_agent` gets fading (`--distortions awgn fading`, the default in `train_netztest.py` line 32) —
arbitrary which branch gets which, the point is just that they're *different*, so neither branch is
training on identical inputs to the other.

---

## 4. The KDCL math: `src/kdcl.py`

This is the part worth being able to reproduce on a whiteboard. `minlogit()` (lines 17-24):
```python
def minlogit(logits, labels):
    # logits: (m=2 branches, batch, 3 classes)
    m, batch, n_classes = logits.shape
    target_logit = logits.gather(2, labels.view(1, batch, 1).expand(m, batch, 1))
    normalized = logits - target_logit          # each branch's logits, shifted so true-class = 0
    teacher_logit, _ = normalized.min(dim=0)     # element-wise min ACROSS the 2 branches
    return teacher_logit
```
Concretely, for one example with true label `y=2` (High) and, say, `logits_3G = [0.1, 0.4, 0.9]`,
`logits_4G = [-0.3, 0.2, 1.5]`:
- `normalized_3G = [0.1-0.9, 0.4-0.9, 0.9-0.9] = [-0.8, -0.5, 0.0]`
- `normalized_4G = [-0.3-1.5, 0.2-1.5, 1.5-1.5] = [-1.8, -1.3, 0.0]`
- `teacher = min([-0.8,-0.5,0.0], [-1.8,-1.3,0.0]) = [-1.8, -1.3, 0.0]`

Notice the true class (index 2) is *always* exactly 0 in both, so the min there is always 0 too —
the teacher never penalizes the correct answer. For the wrong classes, it picked 4G's more negative
(= more confidently *wrong-class-suppressing*) values here — in general it takes whichever branch
was more skeptical of each specific wrong class. That's "never worse than either branch alone" made
concrete.

`naive()` (lines 7-14) and `linear()` (lines 27-42) are the two alternatives — `naive` just copies
whichever branch had the lowest cross-entropy loss on that exact example; `linear` learns a blend
weight per branch via a small internal gradient-descent loop (`n_iters=50`) each call. **Only
`minlogit` has been used in every run so far** — confirmed by grepping `ensemble=` in every
`train_log.txt` under `experiments/`, all say `minlogit`. `linear` exists and is wired up
(`ENSEMBLE_METHODS` dict, line 45-49) but hasn't been tried on this dataset yet.

`teacher_logit_for_mode()` in `train_netztest.py` (lines 66-77) is what actually calls this per
batch, and also implements DML's version in the `mode == "dml"` branch:
```python
if mode == "dml":
    probs = F.softmax(logits_stack, dim=-1)
    peer_targets = [torch.cat([probs[:i], probs[i+1:]]).mean(dim=0) for i in range(m)]
    return torch.stack(peer_targets).detach()
```
For branch `i`, its DML target is just the *other* branch's softmax probabilities directly — no
normalization, no min, no protection. That's the entire difference between DML and KDCL in one
side-by-side comparison.

---

## 5. Loss computation: `compute_losses()` in `train_netztest.py` (lines 80-94)

```python
def compute_losses(mode, T, alpha, logits, labels, teacher):
    for i in range(len(logits)):
        ce_loss = F.cross_entropy(logits[i], labels)
        if mode == "vanilla":
            loss = ce_loss
        elif mode == "dml":
            kd_loss = F.kl_div(F.log_softmax(logits[i]/T, -1), teacher[i], reduction="batchmean") * T*T
            loss = (1-alpha)*ce_loss + alpha*kd_loss
        else:  # kdcl
            kd_loss = F.kl_div(F.log_softmax(logits[i]/T, -1), F.softmax(teacher/T, -1), reduction="batchmean") * T*T
            loss = (1-alpha)*ce_loss + alpha*kd_loss
```
`T` (temperature) softens the probability distribution before comparing — dividing logits by `T>1`
before softmax makes the distribution flatter/less confident, which is standard in all distillation
methods (it gives the KL-divergence term a smoother gradient signal). `T*T` rescales the resulting
loss magnitude back up to compensate (a standard correction from the original Hinton distillation
paper). `alpha` is the tuning knob we swept: `alpha=0.5` (default, inherited from the CIFAR reference
code) vs `alpha=0.1-0.2` (found to work much better for this 3-class task).

---

## 6. The actual loop: `train_one_epoch()`, `evaluate()`, `main()`

`train_one_epoch()` (lines 97-119) — one pass over all training batches:
```python
for views, labels in loader:
    xs = [views[:, i, :] for i in range(len(models))]      # split the two distorted views apart
    logits = [model(x) for model, x in zip(models, xs)]    # forward pass, each branch independently
    logits_stack = torch.stack(logits)
    teacher = teacher_logit_for_mode(args.mode, args.ensemble, logits_stack, labels)
    losses = compute_losses(args.mode, args.T, args.alpha, logits, labels, teacher)
    for i, (opt, loss) in enumerate(zip(optimizers, losses)):
        opt.zero_grad()
        loss.backward(retain_graph=(i < len(models) - 1))   # keep graph alive until the last branch's backward
        opt.step()                                          # update ONLY this branch's weights
```
Each branch has its **own separate optimizer** (`optimizers[i]`, created in `main()`, line 159) —
`opt.step()` only ever touches that one model's parameters. There is no shared weight anywhere
between the two networks; the only thing that ever crosses between them is the `teacher` tensor,
and it's `.detach()`-ed (no gradient flows through it into the other branch).

`evaluate()` (lines 122-135) does a forward pass only (`@torch.no_grad()` on line 122, so no
gradients are even computed) over the validation set, with `model.eval()` called first (line 125) —
this switches off Dropout and freezes BatchNorm's running statistics, so evaluation is deterministic
rather than using the randomized training-mode behavior.

`main()` (lines 138-184) ties it together: loop over `args.epochs` (50), call `train_one_epoch()`
then `evaluate()` each time, and whenever a branch's validation accuracy beats its own previous best
(line 171: `if val_acces[i] > best_acc[i]`), save that branch's weights to
`experiments/.../<branch>/best.pth` (lines 172-176). After all 50 epochs, print the best value each
branch ever reached (lines 183-184) — that's the number reported in `progress_log.md`.

---

## Training vs. validation vs. test — resolving your question

**We have already done training *and* validation together** — they are not sequential phases you
still owe yourself, they happen interleaved, every single epoch:
- **Training** = `train_one_epoch()` — the model sees data, computes a loss, updates its weights.
- **Validation** = `evaluate()`, called right after, on a slice of data (`val_df`, from
  `time_blocked_split`) that was **never used to update any weights**. Its only job is to tell us
  "how good is the model *right now*, on data it hasn't memorized" — and specifically, to decide
  which epoch's weights are worth keeping (`best.pth`).

So "training" as a whole already includes validation monitoring baked in. What you might actually
be thinking of as a separate next phase is one of these two different things:

1. **The held-out test set** (`time_blocked_split`'s third output, `test_df` — currently unused
   anywhere in `train_netztest.py`). Validation accuracy is used *during* training to pick the best
   checkpoint, which technically makes it slightly "used" in a soft sense (you tuned against it).
   The test set is never touched until the very end, for one final, completely unbiased number.
   **We have not done this yet on the new dataset.**
2. **The RAT-selection inference stage** (what `src/infer.py` did for the old dataset) — a
   completely different kind of "validation": not "is the model accurate," but "if you actually used
   this model's predictions to recommend a RAT in practice, would that recommendation be any good?"
   **This also has not been built for the new dataset yet.**

Both are legitimate next steps; they're just not what "training" already covered.
