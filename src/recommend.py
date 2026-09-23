"""RAT-recommendation inference stage. Given a shared network state, run it through both trained
branches and recommend whichever predicts the better expected load class. Default mode expects
models trained WITHOUT the tech flag (is_<RATS[1]>, e.g. is_5G) as a feature (train.py's default
since 2026-07-28) -- both branches reason from the genuine radio-condition features alone.
`--include_tech_flag` supports the legacy comparison mode: the tech flag was a trained feature
there, so both branches could otherwise "read off" a row's actual serving tech instead of reasoning
independently; `recommend_rat` overrides it to each branch's own native value before scoring in
that case.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from rat_dataset import load_raw, make_load_label, numeric_feature_columns, time_blocked_split, LABEL_COL, LOAD_COL
from shared_data import fit_scaler, SingleViewDataset
from models import build_model
from train import BRANCH_NAMES, RATS

N_LOAD_CLASSES = 3


def load_checkpoint(path: str, model_name: str, in_dim: int, n_classes: int = N_LOAD_CLASSES):
    model = build_model(model_name, in_dim, n_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def normalized_tech_flag_values(feature_cols: list, scaler) -> tuple:
    """Returns (idx, norm_value_when_flag=0, norm_value_when_flag=1) for is_<RATS[1]> in the
    scaled feature space."""
    idx = feature_cols.index("is_" + RATS[1])
    mean, std = scaler
    return idx, float((0.0 - mean[idx]) / std[idx]), float((1.0 - mean[idx]) / std[idx])


@torch.no_grad()
def recommend_rat(models, x: torch.Tensor, tech_flag_idx: int = None, tech_flag_norm_values: tuple = None):
    """Returns (recommended_branch_idx, confidences). If tech_flag_idx is given, each branch is
    scored under its own forced tech identity (needed when the tech flag is a trained feature). If
    the model was trained with --exclude_tech_flag (default), there's no such column to force --
    both branches just see the same radio-condition vector and must reason from that alone."""
    confidences = []
    for i, model in enumerate(models):
        x_branch = x.clone()
        if tech_flag_idx is not None:
            x_branch[tech_flag_idx] = tech_flag_norm_values[i]  # branch 0 = RATS[0], branch 1 = RATS[1]
        logits = model(x_branch.unsqueeze(0))
        probs = F.softmax(logits, dim=-1).squeeze(0)
        # assumes higher class index = better KPI outcome (rat_dataset.py make_load_label usage)
        expected_quality = (probs * torch.arange(probs.numel(), dtype=probs.dtype)).sum()
        confidences.append(expected_quality.item())
    best_idx = int(torch.tensor(confidences).argmax())
    return best_idx, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs=2, required=True,
                         help=f"paths to best.pth for [{BRANCH_NAMES[0]}, {BRANCH_NAMES[1]}]")
    parser.add_argument("--model_names", type=str, nargs=2, default=["rat_mlp_small", "rat_mlp_large"])
    parser.add_argument("--n_samples", type=int, default=0, help="0 = evaluate the full held-out test split")
    parser.add_argument("--include_tech_flag", action="store_true",
                         help="match a model trained with train.py's --include_tech_flag (legacy)")
    args = parser.parse_args()

    df = load_raw()
    feature_cols = numeric_feature_columns(df, include_tech_flag=args.include_tech_flag)
    df[LABEL_COL] = make_load_label(df, load_col=LOAD_COL, n_classes=N_LOAD_CLASSES)
    train_df, _, test_df = time_blocked_split(df, order_cols=("time_utc",))
    scaler = fit_scaler(train_df, feature_cols)
    ds = SingleViewDataset(test_df, feature_cols, test_df[LABEL_COL], scaler=scaler)

    models = [load_checkpoint(ckpt, name, len(feature_cols))
              for ckpt, name in zip(args.checkpoints, args.model_names)]
    if args.include_tech_flag:
        tech_flag_idx, norm_0, norm_1 = normalized_tech_flag_values(feature_cols, scaler)
    else:
        tech_flag_idx, norm_0, norm_1 = None, None, None

    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    agree = 0
    high_load_rows, high_load_agree = 0, 0
    recommended_counts = {RATS[0]: 0, RATS[1]: 0}
    for i in range(n):
        idx = test_df.index[i]
        x, _ = ds[i]
        best_idx, confidences = recommend_rat(
            models, x, tech_flag_idx, (norm_0, norm_1) if tech_flag_idx is not None else None
        )
        actual_rat = test_df.loc[idx, "cat_technology"]
        actual_load = test_df.loc[idx, LABEL_COL]
        recommended = BRANCH_NAMES[best_idx].replace("_agent", "")
        recommended_counts[recommended] += 1
        matched = recommended == actual_rat
        agree += int(matched)
        if actual_load == N_LOAD_CLASSES - 1:  # top class actually achieved -- did we recommend the RAT that got it?
            high_load_rows += 1
            high_load_agree += int(matched)
        if n <= 20:
            print(f"row {idx}: recommend={recommended} (confidences={confidences}) actual_tech={actual_rat}")

    majority_rat = test_df["cat_technology"].value_counts(normalize=True).idxmax()
    majority_baseline = test_df["cat_technology"].value_counts(normalize=True).max()
    high_load_agreement = high_load_agree / high_load_rows if high_load_rows else None

    summary_lines = [
        f"Test set size: {n}",
        f"Recommendation distribution: {recommended_counts} "
        f"(a degenerate/constant recommender would show one value at ~{n})",
        f"Agreement with actual serving RAT: {agree}/{n} = {agree/n:.1%} (proxy measure only)",
        f"Majority-class baseline (always recommend '{majority_rat}'): {majority_baseline:.1%} "
        f"-- agreement above this bar is informative, at/below it is not",
    ]
    if high_load_rows:
        summary_lines.append(
            f"Agreement specifically on rows that actually achieved the top load class: "
            f"{high_load_agree}/{high_load_rows} = {high_load_agreement:.1%}"
        )
    print("\n" + "\n".join(summary_lines))

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "recommender_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\nSaved results/recommender_summary.txt")


if __name__ == "__main__":
    main()
