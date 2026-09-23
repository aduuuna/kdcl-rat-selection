"""Runs every comparison method across multiple seeds, back to back, via subprocess calls to
train.py -- Objective 1 (repeated-seed confirmation) and Objective 2 (KDCL vs >=3 other
methods) in one command. Each call gets its own fresh timestamped experiment folder, so re-running
never overwrites a previous run.

Usage:
    python src/run_seeds.py --epochs 5                 # smoke test first -- do this before --epochs 50
    python src/run_seeds.py                             # default: 3 seeds x 5 methods, 50 epochs
    python src/run_seeds.py --seeds 1 2 3 4 5            # 5 seeds
    python src/run_seeds.py --offline_kd_teacher experiments/netztest_vanilla/<timestamp>
                                                          # add offline_kd once a vanilla run exists
"""
import argparse
import subprocess
import sys

METHODS = [
    ("vanilla",     ["--mode", "vanilla"]),
    ("dml",         ["--mode", "dml"]),
    ("kdcl",        ["--mode", "kdcl", "--ensemble", "minlogit", "--T", "1.0", "--alpha", "0.1"]),
    ("kdcl_linear", ["--mode", "kdcl", "--ensemble", "linear", "--T", "1.0", "--alpha", "0.1"]),
    ("one",         ["--mode", "one"]),
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512,
                   help="512 default (not train.py's own 64) -- the vectorized "
                        "FastMultiViewLoader (2026-09-23) makes larger batches cheap; measured "
                        "~7.7s/epoch at 512 on the 200K-row 4G/5G set vs ~34s/epoch at 64.")
    p.add_argument("--methods", type=str, nargs="+", default=None,
                   help="subset of method labels to run (see METHODS), default: all")
    p.add_argument("--offline_kd_teacher", type=str, default=None,
                   help="experiment dir of an already-trained vanilla run, to also run offline_kd")
    return p.parse_args()


def main():
    args = get_args()
    methods = list(METHODS)
    if args.methods:
        methods = [(label, extra) for label, extra in methods if label in args.methods]
    if args.offline_kd_teacher:
        methods.append(("offline_kd", ["--mode", "offline_kd", "--teacher_ckpt", args.offline_kd_teacher]))

    total = len(methods) * len(args.seeds)
    done = 0
    for label, extra_args in methods:
        for seed in args.seeds:
            done += 1
            cmd = [sys.executable, "src/train.py", *extra_args,
                   "--epochs", str(args.epochs), "--seed", str(seed), "--print_freq", str(args.epochs),
                   "--batch_size", str(args.batch_size)]
            print(f"\n=== [{done}/{total}] {label}, seed={seed} ===")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

    print("\nAll runs finished. Now run:  python src/evaluate.py")
    if not args.offline_kd_teacher:
        print("\nTo also include offline_kd, find a vanilla run folder under "
              "experiments/netztest_vanilla/, then re-run:")
        print("  python src/run_seeds.py --offline_kd_teacher experiments/netztest_vanilla/<timestamp> "
              "--seeds " + " ".join(map(str, args.seeds)))


if __name__ == "__main__":
    main()