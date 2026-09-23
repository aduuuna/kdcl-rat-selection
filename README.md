# KDCL for RAT Selection

Final year project, Department of Computer Science, University of Ghana.

## What this is about

Phones today often sit inside range of more than one radio network at once, say a 4G tower and a
newer 5G one. Deciding which one to actually use for a given moment is normally handled by fairly
blunt rules based on signal strength thresholds. This project asks whether a pair of small neural
networks, one for each radio technology, can learn to do this better by training together and
sharing what they learn, rather than being trained alone or trained by copying a single fixed
teacher.

The specific method is called Online Knowledge Distillation via Collaborative Learning (KDCL),
originally published for image classification (Guo et al., CVPR 2020). Instead of one big
pretrained model teaching a smaller one, KDCL lets a group of models train side by side and builds
a "teacher" signal on the fly out of their combined predictions, using whichever ensembling
strategy is least likely to let a weak model drag a strong one down. That negative dragging-down
effect is a real, documented failure mode of an earlier method called Deep Mutual Learning, and
part of what this project checks is whether KDCL actually avoids it once the target domain is
network telemetry instead of pictures.

Two agents are trained here: a smaller MLP that plays the role of a 4G radio agent, and a larger
one for 5G, since 5G genuinely has more headroom and more variable conditions to reason about. Both
agents are trained on the same real, anonymised drive test measurements published by RTR NetzTest
(the Austrian telecoms regulator), covering 2024 and 2025. They are compared under five training
setups: independently trained baselines, Deep Mutual Learning, two variants of KDCL, an adapted
version of On the Fly Native Ensemble, and classic offline distillation. Every run is repeated
across multiple random seeds so the comparison isn't resting on a single lucky (or unlucky) run.

## Repository layout

- `src/rat_dataset.py`: loads the RTR NetzTest data, builds the feature set and the load class
  label, handles the train/validation/test split (kept in chronological order, not shuffled,
  since network conditions drift over time).
- `src/shared_data.py`: small reusable pieces (the dataset classes, the feature scaler, the
  chronological splitter) that both the current pipeline and earlier experiments have relied on.
- `src/distortions.py`: the noise and fading perturbations used for Invariant Collaborative
  Learning, so each agent sees a slightly different, more realistic version of the same reading.
- `src/models.py`: the two MLP architectures and the small gating network used by the ONE
  comparison method.
- `src/kdcl.py`: the actual ensembling maths behind KDCL (naive, min logit, and linear variants).
- `src/train.py`: the training loop itself, one script covering all five methods via a `--mode`
  flag.
- `src/run_seeds.py`: runs a full method across several seeds back to back.
- `src/evaluate.py`: loads every saved checkpoint, scores it on the held out test set, and reports
  mean and standard deviation per method.
- `src/recommend.py`: takes two trained agents and turns their predictions into an actual "use 4G"
  or "use 5G" recommendation, then checks how often that would have matched what really happened.
- `src/extract_4g5g.py` and `src/subsample_4g5g.py`: the one-off scripts used to turn the full,
  multi year RTR NetzTest archive into the smaller training file this project actually uses.
- `run_experiments.ipynb`: a thin notebook wrapper around the scripts above, useful on Google
  Colab where a notebook is the natural way to run things.
- `experiments/`: the actual logs and checkpoints from real training runs, kept as evidence rather
  than regenerated claims.
- `docs/paper/`: the original KDCL paper this project builds on.

## The data

The training file is about 25MB and isn't included in this repository (see `.gitignore`), mostly
because the original source archive it comes from is over 12GB and clearly doesn't belong in git.
Download `netztest-4g5g-2024-2025-sample200k.csv` from
[Google Drive](https://drive.google.com/drive/folders/1YdErcQsabx25vtBv4kJrpjcldw3erNXM?usp=sharing)
and place it at `data/raw/netztest-4g5g-2024-2025-sample200k.csv`. On Colab, copy it into your own
Drive from that same link, then set the `KDCL_RAT_CSV` environment variable to point at it, which
the notebook does for you in its first cell.

## Running it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Smoke test first, a couple of epochs just to check everything runs on your machine:

```
python src/train.py --mode kdcl --epochs 2
```

Then run the full comparison, which trains all five methods across three seeds:

```
python src/run_seeds.py --epochs 50
```

Once that finishes, score everything on the held out test set:

```
python src/evaluate.py
```

Or open `run_experiments.ipynb` and run it top to bottom, which does the same thing but also
produces a results table and a chart.

## Contact

This is a research project, and questions, corrections, or collaboration interest are welcome.

- Joy Owusu Ansah (researcher), owusujoyansah@gmail.com
- Mr. Julius Yaw Ludu (supervisor), jyludu@ug.edu.gh

## Where this stands

This is active, ongoing work. Earlier phases of this project used a different technology pair
(3G and 4G) on an older data slice, and those results are kept in `experiments/` and
`docs/notes/` for the record rather than deleted. The current phase, 4G versus 5G on 2024 to 2025
data, is the one this README describes and the one still being run and written up.
