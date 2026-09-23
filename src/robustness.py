"""Robustness-to-noise evaluation: injects increasing AWGN into the held-out TEST split and
measures accuracy degradation per method (mirrors the KDCL paper's Figure 4, and this project's
thesis Table 4.3 / Figure 4.2). Rewritten 2026-09-23 -- the previous version of this script
pointed at checkpoint paths from a 2026-07-16 naming scheme that no longer exists; this one
auto-discovers checkpoints the same way evaluate.py does, and averages across every seed found
per mode/branch rather than assuming exactly one checkpoint per mode.
"""
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
RESULTS_DIR = "results"
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

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
    train_df, _, test_df = time_blocked_split(df, order_cols=("time_utc",))
    scaler = fit_scaler(train_df, feature_cols)

    test_ds = SingleViewDataset(test_df, feature_cols, test_df[LABEL_COL], scaler=scaler)
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    rng = torch.Generator().manual_seed(0)
    rows = []

    for mode_dir, branches in MODE_BRANCHES.items():
        runs = find_runs(mode_dir)
        if not runs:
            continue
        for branch, arch in branches.items():
            ckpts = [r for r in runs if os.path.exists(os.path.join(r, branch, "best.pth"))]
            if not ckpts:
                continue
            models = [load_checkpoint(r, branch, arch, len(feature_cols)) for r in ckpts]
            for sigma in NOISE_LEVELS:
                X_noisy = X_test + torch.randn(X_test.shape, generator=rng) * sigma
                accs = []
                with torch.no_grad():
                    for model in models:
                        accs.append(accuracy(model(X_noisy), y_test)[0].item())
                mean_acc = sum(accs) / len(accs)
                std_acc = stats.pstdev(accs) if len(accs) > 1 else 0.0
                rows.append({"mode": mode_dir, "branch": branch, "sigma": sigma,
                             "n_runs": len(accs), "acc_mean": mean_acc, "acc_std": std_acc})
                print(f"{mode_dir:<20}{branch:<10}sigma={sigma:<5}acc={mean_acc:.2f}+/-{std_acc:.2f} "
                      f"(n={len(accs)})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "robustness.csv")
    with open(out_path, "w") as f:
        f.write("mode,branch,sigma,n_runs,acc_mean,acc_std\n")
        for r in rows:
            f.write(f"{r['mode']},{r['branch']},{r['sigma']},{r['n_runs']},"
                     f"{r['acc_mean']:.4f},{r['acc_std']:.4f}\n")
    print(f"\nSaved {out_path} ({len(rows)} rows).")
    return rows


if __name__ == "__main__":
    main()
