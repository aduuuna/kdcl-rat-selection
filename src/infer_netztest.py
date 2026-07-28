"""RAT-recommendation inference stage for the netztest (3G/4G) dataset. Given a shared network
state, run it through both trained branches and recommend whichever predicts the better expected
load class. Built with the confound fix from the start (see build_explanation.md and plan.md) --
`is_4G` is a trained feature, so both branches could otherwise "read off" a row's actual serving
tech instead of reasoning independently. `recommend_rat` overrides `is_4G` to each branch's own
native value before scoring.
"""
import argparse

import torch
import torch.nn.functional as F

from netztest_data import load_raw, make_load_label, numeric_feature_columns, time_blocked_split, LABEL_COL, LOAD_COL
from data import fit_scaler, SingleViewDataset
from models import build_model
from train_netztest import BRANCH_NAMES, RATS

N_LOAD_CLASSES = 3


def load_checkpoint(path: str, model_name: str, in_dim: int, n_classes: int = N_LOAD_CLASSES):
    model = build_model(model_name, in_dim, n_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def normalized_is_4g_values(feature_cols: list, scaler) -> tuple:
    """Returns (idx, norm_value_for_is_4G=0, norm_value_for_is_4G=1) in the scaled feature space."""
    idx = feature_cols.index("is_" + RATS[1])
    mean, std = scaler
    return idx, float((0.0 - mean[idx]) / std[idx]), float((1.0 - mean[idx]) / std[idx])


@torch.no_grad()
def recommend_rat(models, x: torch.Tensor, is_4g_idx: int, is_4g_norm_values: tuple):
    """Returns (recommended_branch_idx, confidences), each branch scored under its own tech identity."""
    confidences = []
    for i, model in enumerate(models):
        x_branch = x.clone()
        x_branch[is_4g_idx] = is_4g_norm_values[i]  # branch 0 = 3G (is_4G=0), branch 1 = 4G (is_4G=1)
        logits = model(x_branch.unsqueeze(0))
        probs = F.softmax(logits, dim=-1).squeeze(0)
        # assumes higher class index = better KPI outcome (netztest_data.py make_load_label usage)
        expected_quality = (probs * torch.arange(probs.numel(), dtype=probs.dtype)).sum()
        confidences.append(expected_quality.item())
    best_idx = int(torch.tensor(confidences).argmax())
    return best_idx, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs=2, required=True,
                         help="paths to best.pth for [3G_agent, 4G_agent]")
    parser.add_argument("--model_names", type=str, nargs=2, default=["rat_mlp_small", "rat_mlp_large"])
    parser.add_argument("--n_samples", type=int, default=0, help="0 = evaluate the full held-out test split")
    args = parser.parse_args()

    df = load_raw()
    feature_cols = numeric_feature_columns(df)
    df[LABEL_COL] = make_load_label(df, load_col=LOAD_COL, n_classes=N_LOAD_CLASSES)
    train_df, _, test_df = time_blocked_split(df, order_cols=("time_utc",))
    scaler = fit_scaler(train_df, feature_cols)
    ds = SingleViewDataset(test_df, feature_cols, test_df[LABEL_COL], scaler=scaler)

    models = [load_checkpoint(ckpt, name, len(feature_cols))
              for ckpt, name in zip(args.checkpoints, args.model_names)]
    is_4g_idx, norm_0, norm_1 = normalized_is_4g_values(feature_cols, scaler)

    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    agree = 0
    high_load_rows, high_load_agree = 0, 0
    for i in range(n):
        idx = test_df.index[i]
        x, _ = ds[i]
        best_idx, confidences = recommend_rat(models, x, is_4g_idx, (norm_0, norm_1))
        actual_rat = test_df.loc[idx, "cat_technology"]
        actual_load = test_df.loc[idx, LABEL_COL]
        recommended = BRANCH_NAMES[best_idx].replace("_agent", "")
        matched = recommended == actual_rat
        agree += int(matched)
        if actual_load == N_LOAD_CLASSES - 1:  # top class actually achieved -- did we recommend the RAT that got it?
            high_load_rows += 1
            high_load_agree += int(matched)
        if n <= 20:
            print(f"row {idx}: recommend={recommended} (confidences={confidences}) actual_tech={actual_rat}")

    majority_rat = test_df["cat_technology"].value_counts(normalize=True).idxmax()
    majority_baseline = test_df["cat_technology"].value_counts(normalize=True).max()

    print(f"\nTest set size: {n}")
    print(f"Agreement with actual serving RAT: {agree}/{n} = {agree/n:.1%} (proxy measure only)")
    print(f"Majority-class baseline (always recommend '{majority_rat}'): {majority_baseline:.1%} "
          f"-- agreement above this bar is informative, at/below it is not")
    if high_load_rows:
        print(f"Agreement specifically on rows that actually achieved the top load class: "
              f"{high_load_agree}/{high_load_rows} = {high_load_agree/high_load_rows:.1%}")


if __name__ == "__main__":
    main()
