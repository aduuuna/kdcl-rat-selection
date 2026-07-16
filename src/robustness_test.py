"""Phase 6 robustness test: inject increasing AWGN noise into the validation set at test time and
compare accuracy degradation across vanilla/DML/KDCL trained checkpoints (mirrors the paper's
Figure 4). This is the real test of whether ICL (distortion during training) helps -- its purpose
is robustness to noisy inputs, not clean-validation accuracy (which the ablation in progress_log.md
already showed gets slightly worse with distortion enabled).
"""
import torch

from data import load_raw, make_load_label, numeric_feature_columns, time_blocked_split, LABEL_COL, fit_scaler
from models import build_model
from utils import accuracy

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

CHECKPOINTS = {
    "vanilla": "experiments/vanilla_rat_mlp_small_rat_mlp_large/2026-07-16-02-58",
    "dml": "experiments/dml_rat_mlp_small_rat_mlp_large/2026-07-16-03-01",
    "kdcl_tuned": "experiments/kdcl_rat_mlp_small_rat_mlp_large/2026-07-16-03-10",
    "kdcl_tuned_no_icl": "experiments/kdcl_rat_mlp_small_rat_mlp_large/2026-07-16-03-14",
}
BRANCH_MODEL_NAMES = {"4G_agent": "rat_mlp_small", "5G_agent": "rat_mlp_large"}


def load_model(ckpt_dir, branch, in_dim, n_classes=3):
    model = build_model(BRANCH_MODEL_NAMES[branch], in_dim, n_classes)
    state = torch.load(f"{ckpt_dir}/{branch}/best.pth", map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main():
    df = load_raw()
    feature_cols = numeric_feature_columns(df)
    df[LABEL_COL] = make_load_label(df)
    train_df, val_df, _ = time_blocked_split(df)
    mean, std = fit_scaler(train_df, feature_cols)

    X_raw = val_df[feature_cols].fillna(0.0).to_numpy(dtype="float32")
    X = torch.tensor((X_raw - mean) / std)
    y = torch.tensor(val_df[LABEL_COL].to_numpy(dtype="int64"))
    rng = torch.Generator().manual_seed(0)

    results = {mode: {branch: [] for branch in BRANCH_MODEL_NAMES} for mode in CHECKPOINTS}

    for mode, ckpt_dir in CHECKPOINTS.items():
        models = {branch: load_model(ckpt_dir, branch, len(feature_cols)) for branch in BRANCH_MODEL_NAMES}
        for sigma in NOISE_LEVELS:
            X_noisy = X + torch.randn(X.shape, generator=rng) * sigma
            for branch, model in models.items():
                with torch.no_grad():
                    acc = accuracy(model(X_noisy), y)[0].item()
                results[mode][branch].append(acc)

    header = f"{'sigma':>6}"
    for mode in CHECKPOINTS:
        for branch in BRANCH_MODEL_NAMES:
            header += f"  {mode}/{branch:>12}"
    print(header)

    for i, sigma in enumerate(NOISE_LEVELS):
        row = f"{sigma:>6}"
        for mode in CHECKPOINTS:
            for branch in BRANCH_MODEL_NAMES:
                row += f"  {results[mode][branch][i]:>{len(mode) + 15}.2f}"
        print(row)


if __name__ == "__main__":
    main()
