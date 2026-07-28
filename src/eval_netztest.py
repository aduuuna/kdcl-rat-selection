"""Final, unbiased evaluation on the held-out TEST split (time_blocked_split's third slice --
never used during training or to pick a checkpoint, unlike the validation numbers in
progress_log.md). Loads saved checkpoints from specific experiment runs and reports test accuracy
per branch per mode, for direct comparison against the validation numbers.
"""
import torch

from netztest_data import load_raw, make_load_label, numeric_feature_columns, time_blocked_split, LABEL_COL, LOAD_COL
from data import fit_scaler, SingleViewDataset
from models import build_model
from utils import accuracy

BRANCH_MODEL_NAMES = {"3G_agent": "rat_mlp_small", "4G_agent": "rat_mlp_large"}

# Current baseline (no tech flag -- default since 2026-07-28). See plan.md / progress_log.md.
CHECKPOINTS = {
    "vanilla": "experiments/netztest_vanilla/2026-07-28-14-27",
    "dml": "experiments/netztest_dml/2026-07-28-14-30",
    "kdcl_tuned": "experiments/netztest_kdcl/2026-07-28-14-34",  # T=1.0, alpha=0.1
}


def load_checkpoint(ckpt_dir, branch, in_dim, n_classes=3):
    model = build_model(BRANCH_MODEL_NAMES[branch], in_dim, n_classes)
    state = torch.load(f"{ckpt_dir}/{branch}/best.pth", map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main():
    df = load_raw()
    feature_cols = numeric_feature_columns(df, include_tech_flag=False)  # matches how CHECKPOINTS were trained
    df[LABEL_COL] = make_load_label(df, load_col=LOAD_COL, n_classes=3)
    train_df, val_df, test_df = time_blocked_split(df, order_cols=("time_utc",))
    scaler = fit_scaler(train_df, feature_cols)

    val_ds = SingleViewDataset(val_df, feature_cols, val_df[LABEL_COL], scaler=scaler)
    test_ds = SingleViewDataset(test_df, feature_cols, test_df[LABEL_COL], scaler=scaler)

    X_val = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    y_val = torch.tensor([val_ds[i][1] for i in range(len(val_ds))])
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    print(f"val set size: {len(y_val)}, test set size: {len(y_test)}\n")
    print(f"{'mode':<14}{'branch':<10}{'val_acc':>10}{'test_acc':>10}")
    for mode, ckpt_dir in CHECKPOINTS.items():
        for branch in BRANCH_MODEL_NAMES:
            model = load_checkpoint(ckpt_dir, branch, len(feature_cols))
            with torch.no_grad():
                val_acc = accuracy(model(X_val), y_val)[0].item()
                test_acc = accuracy(model(X_test), y_test)[0].item()
            print(f"{mode:<14}{branch:<10}{val_acc:>10.2f}{test_acc:>10.2f}")


if __name__ == "__main__":
    main()
