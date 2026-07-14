"""RAT-recommendation inference stage: runs a network-state vector through both trained branches
and recommends whichever predicts the better KPI outcome. Ground truth for "correct
recommendation" is still an open question (plan.md) -- agreement with actual NetworkTech below
is a proxy, not a true correctness measure."""
import argparse

import torch
import torch.nn.functional as F

from data import load_raw, make_load_label, numeric_feature_columns, SingleViewDataset
from models import build_model
from train import BRANCH_NAMES, N_LOAD_CLASSES


def load_checkpoint(path: str, model_name: str, in_dim: int, n_classes: int = N_LOAD_CLASSES):
    model = build_model(model_name, in_dim, n_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def recommend_rat(models, x: torch.Tensor):
    """Returns (recommended_branch_idx, confidences) for a single shared state vector."""
    confidences = []
    for model in models:
        logits = model(x.unsqueeze(0))
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
    parser.add_argument("--model_names", type=str, nargs=2, default=["rat_mlp", "rat_mlp"])
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    df = load_raw()
    df["load_class"] = make_load_label(df)
    feature_cols = numeric_feature_columns(df)
    ds = SingleViewDataset(df, feature_cols, df["load_class"])

    models = [load_checkpoint(ckpt, name, len(feature_cols))
              for ckpt, name in zip(args.checkpoints, args.model_names)]

    agree = 0
    for idx in range(min(args.n_samples, len(ds))):
        x, _ = ds[idx]
        best_idx, confidences = recommend_rat(models, x)
        actual_rat = df.iloc[idx]["NetworkTech"]
        recommended = BRANCH_NAMES[best_idx].replace("_agent", "")
        agree += int(recommended == actual_rat)
        print(f"row {idx}: recommend={recommended} (confidences={confidences}) actual_tech={actual_rat}")

    print(f"\nAgreement with actual NetworkTech in use: {agree}/{min(args.n_samples, len(ds))} (proxy measure only)")


if __name__ == "__main__":
    main()
