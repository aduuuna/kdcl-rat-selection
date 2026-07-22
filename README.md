# KDCL for RAT Selection

Final year project (Dept. of Computer Science, University of Ghana). Adapts **KDCL — Online Knowledge
Distillation via Collaborative Learning** (CVPR 2020) from image classifiers to heterogeneous wireless
RAT agents (4G/5G), trained collaboratively on real drive-test telemetry, with a downstream RAT-selection
recommendation stage.

See [plan.md](plan.md) for the full phased roadmap,
[docs/notes/progress_log.md](docs/notes/progress_log.md) for the running log of what's been done,
findings, and decisions made along the way, and
[docs/notes/writeup.md](docs/notes/writeup.md) for the thesis-chapter-structured write-up (Phase 8),
and [docs/notes/explanation.md](docs/notes/explanation.md) for a plain-language walkthrough of the
whole codebase and what it does.

## Layout
- `src/` — data pipeline, distortions, models, KDCL ensemble methods, training/inference (this project's code)
- `reference/` — the original image-classification KDCL reimplementation (CIFAR-100), kept as-is for reference
- `data/raw/` — source dataset (`urban_dataset_processed.xlsx`)
- `data/processed/` — generated splits/caches (gitignored except this placeholder)
- `docs/paper/` — the KDCL CVPR 2020 paper
- `docs/notes/` — derivation notes, `progress_log.md` (running log), `writeup.md` (thesis-structured draft)
- `experiments/` — run configs and results, tracked in git (per-run `train_log.txt` + checkpoints as evidence)
- `notebooks/` — exploratory data analysis

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
`data/raw/urban_dataset_processed.xlsx` — 22,690 rows of real drive-test telemetry. Confirmed RAT split:
**5G: 14,845 rows, 4G: 7,844 rows** (no 3G/2G/6G present) — the two KDCL students are the 4G-agent and
5G-agent.
