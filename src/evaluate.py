"""Final, unbiased evaluation on the held-out TEST split. Auto-discovers every timestamped run
folder under experiments/netztest_<mode>/ and reports mean +/- std across seeds/runs when more
than one exists per mode -- this is what answers "is the KDCL-vs-vanilla gap stable across seeds"
(Objective 1) and gives a single comparison table across all methods (Objective 2)."""
import glob
import os
import statistics as stats

import torch

from rat_dataset import (
    load_raw, make_load_label, numeric_feature_columns, time_blocked_split, LABEL_COL, LOAD_COL, RATS,
)
from shared_data import fit_scaler, SingleViewDataset
from models import build_model
from utils import accuracy

BRANCH_NAMES = (f"{RATS[0]}_agent", f"{RATS[1]}_agent")

# experiment-folder prefix -> {branch_folder_name: model_arch_name}
# offline_kd only ever produces the student branch (BRANCH_NAMES[0]).
MODE_BRANCHES = {
    "netztest_vanilla":     {BRANCH_NAMES[0]: "rat_mlp_small", BRANCH_NAMES[1]: "rat_mlp_large"},
    "netztest_dml":         {BRANCH_NAMES[0]: "rat_mlp_small", BRANCH_NAMES[1]: "rat_mlp_large"},
    "netztest_kdcl":        {BRANCH_NAMES[0]: "rat_mlp_small", BRANCH_NAMES[1]: "rat_mlp_large"},
    "netztest_kdcl_linear": {BRANCH_NAMES[0]: "rat_mlp_small", BRANCH_NAMES[1]: "rat_mlp_large"},
    "netztest_one":         {BRANCH_NAMES[0]: "rat_mlp_small", BRANCH_NAMES[1]: "rat_mlp_large"},
    "netztest_offline_kd":  {BRANCH_NAMES[0]: "rat_mlp_small"},
}


def find_runs(mode_dir):
    return sorted(d for d in glob.glob(os.path.join("experiments", mode_dir, "*")) if os.path.isdir(d))


def load_checkpoint(ckpt_dir, branch, arch, in_dim, n_classes=3):
    model = build_model(arch, in_dim, n_classes)
    state = torch.load(os.path.join(ckpt_dir, branch, "best.pth"), map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main():
    df = load_raw()
    feature_cols = numeric_feature_columns(df, include_tech_flag=False)
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

    rows = []
    print(f"{'mode':<20}{'branch':<10}{'n_runs':>7}{'val_acc':>18}{'test_acc':>18}")
    for mode_dir, branches in MODE_BRANCHES.items():
        runs = find_runs(mode_dir)
        if not runs:
            continue
        for branch, arch in branches.items():
            val_accs, test_accs = [], []
            for run_dir in runs:
                ckpt_path = os.path.join(run_dir, branch, "best.pth")
                if not os.path.exists(ckpt_path):
                    continue
                model = load_checkpoint(run_dir, branch, arch, len(feature_cols))
                with torch.no_grad():
                    val_accs.append(accuracy(model(X_val), y_val)[0].item())
                    test_accs.append(accuracy(model(X_test), y_test)[0].item())
            if not val_accs:
                continue
            v_mean, t_mean = sum(val_accs) / len(val_accs), sum(test_accs) / len(test_accs)
            v_std = stats.pstdev(val_accs) if len(val_accs) > 1 else 0.0
            t_std = stats.pstdev(test_accs) if len(test_accs) > 1 else 0.0
            print(f"{mode_dir:<20}{branch:<10}{len(val_accs):>7}"
                  f"{v_mean:>11.2f}+/-{v_std:<5.2f}{t_mean:>11.2f}+/-{t_std:<5.2f}")
            rows.append({"mode": mode_dir, "branch": branch, "n_runs": len(val_accs),
                         "val_acc_mean": v_mean, "val_acc_std": v_std,
                         "test_acc_mean": t_mean, "test_acc_std": t_std,
                         "val_accs": val_accs, "test_accs": test_accs})
    return rows


if __name__ == "__main__":
    main()