"""RAT-recommendation inference stage: runs a network-state vector through both trained branches
and recommends whichever predicts the better KPI outcome. Ground truth for "correct
recommendation" is still an open question (plan.md) -- agreement with actual NetworkTech below
is a proxy, not a true correctness measure.

Important: `is_5G` is one of the trained features, so a row's actual serving tech is baked into
its input. Naively feeding the same row to both branches lets each one "read off" which tech was
really in use rather than reasoning independently -- recommend_rat below overrides `is_5G` to each
branch's own native value (0 for 4G_agent, 1 for 5G_agent) before scoring, so the comparison
reflects "how would my tech's model see this state" rather than leaking the real answer.
"""
import argparse

import torch
import torch.nn.functional as F

from data import (
    load_raw, make_load_label, numeric_feature_columns, SingleViewDataset, LABEL_COL,
    time_blocked_split, fit_scaler,
)
from models import build_model
from train import BRANCH_NAMES, N_LOAD_CLASSES


def load_checkpoint(path: str, model_name: str, in_dim: int, n_classes: int = N_LOAD_CLASSES):
    model = build_model(model_name, in_dim, n_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def normalized_is_5g_values(feature_cols: list, scaler) -> tuple:
    """Returns (norm_value_for_is_5G=0, norm_value_for_is_5G=1) in the scaled feature space."""
    idx = feature_cols.index("is_5G")
    mean, std = scaler
    return idx, float((0.0 - mean[idx]) / std[idx]), float((1.0 - mean[idx]) / std[idx])


@torch.no_grad()
def recommend_rat(models, x: torch.Tensor, is_5g_idx: int, is_5g_norm_values: tuple):
    """Returns (recommended_branch_idx, confidences), each branch scored under its own tech identity."""
    confidences = []
    for i, model in enumerate(models):
        x_branch = x.clone()
        x_branch[is_5g_idx] = is_5g_norm_values[i]  # branch 0 = 4G (is_5G=0), branch 1 = 5G (is_5G=1)
        logits = model(x_branch.unsqueeze(0))
        probs = F.softmax(logits, dim=-1).squeeze(0)
        # assumes higher class index = better KPI outcome (data.py make_load_label)
        expected_quality = (probs * torch.arange(probs.numel(), dtype=probs.dtype)).sum()
        confidences.append(expected_quality.item())
    best_idx = int(torch.tensor(confidences).argmax())
    return best_idx, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs=2, required=True,
                         help="paths to best.pth for [4G_agent, 5G_agent]")
    parser.add_argument("--model_names", type=str, nargs=2, default=["rat_mlp_small", "rat_mlp_large"])
    parser.add_argument("--n_samples", type=int, default=0, help="0 = evaluate the full held-out test split")
    args = parser.parse_args()

    df = load_raw()
    feature_cols = numeric_feature_columns(df)
    df[LABEL_COL] = make_load_label(df)
    train_df, _, test_df = time_blocked_split(df)  # test split: untouched by training or hyperparameter tuning
    scaler = fit_scaler(train_df, feature_cols)  # must match the scaler the checkpoints were trained with
    ds = SingleViewDataset(test_df, feature_cols, test_df[LABEL_COL], scaler=scaler)

    models = [load_checkpoint(ckpt, name, len(feature_cols))
              for ckpt, name in zip(args.checkpoints, args.model_names)]
    is_5g_idx, norm_0, norm_1 = normalized_is_5g_values(feature_cols, scaler)

    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    agree = 0
    high_load_rows, high_load_agree = 0, 0
    for i in range(n):
        idx = test_df.index[i]
        x, _ = ds[i]
        best_idx, confidences = recommend_rat(models, x, is_5g_idx, (norm_0, norm_1))
        actual_rat = test_df.loc[idx, "NetworkTech"]
        actual_load = test_df.loc[idx, LABEL_COL]
        recommended = BRANCH_NAMES[best_idx].replace("_agent", "")
        matched = recommended == actual_rat
        agree += int(matched)
        if actual_load == N_LOAD_CLASSES - 1:  # top class actually achieved -- did we recommend the RAT that got it?
            high_load_rows += 1
            high_load_agree += int(matched)
        if n <= 20:
            print(f"row {idx}: recommend={recommended} (confidences={confidences}) actual_tech={actual_rat}")

    majority_rat = test_df["NetworkTech"].value_counts(normalize=True).idxmax()
    majority_baseline = test_df["NetworkTech"].value_counts(normalize=True).max()

    print(f"\nTest set size: {n}")
    print(f"Agreement with actual NetworkTech in use: {agree}/{n} = {agree/n:.1%} (proxy measure only)")
    print(f"Majority-class baseline (always recommend '{majority_rat}'): {majority_baseline:.1%} "
          f"-- agreement above this bar is informative, at/below it is not")
    if high_load_rows:
        print(f"Agreement specifically on rows that actually achieved the top load class: "
              f"{high_load_agree}/{high_load_rows} = {high_load_agree/high_load_rows:.1%}")


if __name__ == "__main__":
    main()
