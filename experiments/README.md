# What's in this folder

Every subfolder here is a real, completed training run, kept as evidence rather than a claim you
have to take on trust. This file explains how to read it, including the `.pth` files, which
aren't human readable on their own.

## Folder layout

```
experiments/
  netztest_vanilla/<timestamp>/       one independently-trained run
  netztest_dml/<timestamp>/           one Deep Mutual Learning run
  netztest_kdcl/<timestamp>/          one KDCL-MinLogit run
  netztest_kdcl_linear/<timestamp>/   one KDCL-Linear run
  netztest_one/<timestamp>/           one adapted On-the-fly Native Ensemble run
  netztest_offline_kd/<timestamp>/    one classic offline distillation run
  _archive_3g4g_thesis_2026-07-28/    earlier phase of this project (3G/4G, kept for the record)
```

Each `<timestamp>` folder is one full run: 50 epochs, one random seed. Multiple timestamp folders
under the same mode are the same method repeated with different seeds, which is what lets
`evaluate.py` report a mean and standard deviation instead of a single number that might just be
luck.

Inside each run folder:

```
netztest_kdcl/2026-09-23-10-56-30/
  train_log.txt              the exact command-line arguments used, plus per-epoch metrics
  4G_agent/best.pth          the trained 4G model, saved at its best validation epoch
  5G_agent/best.pth          the trained 5G model, saved at its best validation epoch
```

(`netztest_offline_kd` runs only produce one model, the student, since the teacher is a frozen
checkpoint borrowed from an existing vanilla run rather than trained fresh.)

## What a `.pth` file actually is

It's a Python dictionary, saved by `torch.save()`, containing exactly three things:

- `epoch`: which training epoch this snapshot is from. Training runs for 50 epochs regardless,
  but only the epoch with the best validation accuracy gets saved, everything after it that didn't
  improve is discarded.
- `acc`: that best validation accuracy.
- `model`: the actual trained network, as a mapping from each layer's name to its numbers
  (weights, biases, and BatchNorm running statistics). This is the real substance of the file,
  typically a few thousand individual floating point numbers, each one adjusted by gradient
  descent over real training data.

You can open any checkpoint in this folder yourself with a few lines of Python:

```python
import torch
ckpt = torch.load("experiments/netztest_kdcl/2026-09-23-10-56-30/4G_agent/best.pth", map_location="cpu")
print("best epoch:", ckpt["epoch"])
print("val accuracy at that epoch:", ckpt["acc"])
for name, tensor in ckpt["model"].items():
    print(name, tuple(tensor.shape))
```

Or load it back into an actual working model and run it on data, which is exactly what
`evaluate.py`, `recommend.py`, and `robustness.py` all do:

```python
import sys; sys.path.insert(0, "src")
from models import build_model
model = build_model("rat_mlp_small", in_dim=17, n_classes=3)
model.load_state_dict(ckpt["model"])
model.eval()
```

## Why this exists instead of just a results table

Numbers in a table or a chat message can be typed by anyone. A checkpoint that loads without
error, runs inference, and produces the same accuracy it claims to, cannot be faked the same way.
That's the point of keeping these files instead of deleting them once the summary numbers are
computed.
