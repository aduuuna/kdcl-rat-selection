"""Phase 6 hyperparameter sweep: grid search over T and alpha for KDCL-MinLogit.
Loads data once and reuses it across the grid (train.py reloads the xlsx per invocation, which
dominates runtime at this dataset size). Reduced epoch budget for the search; re-run the winning
combo with train.py's full 50 epochs to get the final reportable number.
"""
import itertools
from types import SimpleNamespace

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from data import N_LOAD_CLASSES
from models import build_model
from train import build_loaders, train_one_epoch, evaluate

T_GRID = [1.0, 2.0, 4.0, 8.0]
ALPHA_GRID = [0.2, 0.5, 0.8]
SWEEP_EPOCHS = 30
SWEEP_MILESTONES = [18, 24]  # scaled 0.6x from train.py's default [30, 40]/50 schedule


def make_args(T, alpha):
    return SimpleNamespace(
        mode="kdcl", ensemble="minlogit",
        model_names=["rat_mlp_small", "rat_mlp_large"],
        distortions=["awgn", "fading"],
        T=T, alpha=alpha,
        batch_size=64, epochs=SWEEP_EPOCHS, lr=1e-3, weight_decay=1e-4,
        milestones=SWEEP_MILESTONES, gamma=0.1, seed=1, num_workers=0,
    )


def run_one(args, train_loader, val_loader, in_dim):
    models_, optimizers, schedulers = [], [], []
    for name in args.model_names:
        model = build_model(name, in_dim, N_LOAD_CLASSES)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        sched = MultiStepLR(opt, args.milestones, args.gamma)
        models_.append(model)
        optimizers.append(opt)
        schedulers.append(sched)

    best_acc = [-1.0, -1.0]
    for _ in range(args.epochs):
        train_one_epoch(models_, optimizers, train_loader, args)
        _, val_acces = evaluate(models_, val_loader)
        for i in range(2):
            schedulers[i].step()
            best_acc[i] = max(best_acc[i], val_acces[i])
    return best_acc


def main():
    base_args = make_args(T_GRID[0], ALPHA_GRID[0])
    train_loader, val_loader, in_dim = build_loaders(base_args)

    results = []
    for T, alpha in itertools.product(T_GRID, ALPHA_GRID):
        args = make_args(T, alpha)
        acc = run_one(args, train_loader, val_loader, in_dim)
        print(f"T={T} alpha={alpha}: 4G={acc[0]:.2f} 5G={acc[1]:.2f}", flush=True)
        results.append((T, alpha, acc[0], acc[1]))

    print("\n=== Sorted by 5G_agent val acc ===")
    for T, alpha, a4, a5 in sorted(results, key=lambda r: -r[3]):
        print(f"T={T} alpha={alpha}: 4G={a4:.2f} 5G={a5:.2f}")

    print("\n=== Sorted by average val acc ===")
    for T, alpha, a4, a5 in sorted(results, key=lambda r: -(r[2] + r[3])):
        print(f"T={T} alpha={alpha}: 4G={a4:.2f} 5G={a5:.2f} avg={(a4+a5)/2:.2f}")


if __name__ == "__main__":
    main()
