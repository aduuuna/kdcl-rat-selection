"""Multi-branch training loop for the 4G/5G RAT students. Three modes for the baseline
comparisons in plan.md: vanilla (no distillation), dml (mutual peer KL), kdcl (ensemble teacher
logit, default MinLogit)."""
import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from data import (
    load_raw, make_load_label, numeric_feature_columns,
    time_blocked_split, MultiViewDataset, N_LOAD_CLASSES, LABEL_COL, fit_scaler,
)
from distortions import DISTORTION_REGISTRY
from kdcl import ENSEMBLE_METHODS
from models import build_model
from utils import AverageMeter, accuracy

BRANCH_NAMES = ("4G_agent", "5G_agent")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "dml", "kdcl"], default="kdcl")
    parser.add_argument("--ensemble", choices=list(ENSEMBLE_METHODS), default="minlogit")
    parser.add_argument("--model_names", type=str, nargs=2, default=["rat_mlp_small", "rat_mlp_large"],
                         help="one model name per branch, in the order 4G_agent 5G_agent")
    parser.add_argument("--distortions", type=str, nargs=2, default=["awgn", "fading"],
                         help="one distortion per branch, in the order 4G_agent 5G_agent")
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for KD vs CE loss")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[30, 40])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=N_LOAD_CLASSES,
                         help="quantile bins for the load label -- more classes = harder task")
    return parser.parse_args()


def build_loaders(args):
    df = load_raw()
    feature_cols = numeric_feature_columns(df)
    df[LABEL_COL] = make_load_label(df, n_classes=args.n_classes)
    train_df, val_df, _ = time_blocked_split(df)
    scaler = fit_scaler(train_df, feature_cols)

    distortions = [DISTORTION_REGISTRY[args.distortions[i]](seed=args.seed + i) for i in range(2)]
    train_ds = MultiViewDataset(train_df, feature_cols, train_df[LABEL_COL], distortions, scaler=scaler)
    val_ds = MultiViewDataset(val_df, feature_cols, val_df[LABEL_COL], [lambda x: x, lambda x: x], scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, len(feature_cols)


def teacher_logit_for_mode(mode, ensemble_name, logits_stack, labels):
    if mode == "kdcl":
        return ENSEMBLE_METHODS[ensemble_name](logits_stack, labels).detach()
    if mode == "dml":
        m = logits_stack.shape[0]
        probs = F.softmax(logits_stack, dim=-1)
        peer_targets = []
        for i in range(m):
            others = torch.cat([probs[:i], probs[i + 1:]], dim=0)
            peer_targets.append(others.mean(dim=0))
        return torch.stack(peer_targets).detach()
    return None


def compute_losses(mode, T, alpha, logits, labels, teacher):
    losses = []
    for i in range(len(logits)):
        ce_loss = F.cross_entropy(logits[i], labels)
        if mode == "vanilla":
            losses.append(ce_loss)
        elif mode == "dml":
            kd_loss = F.kl_div(F.log_softmax(logits[i] / T, dim=-1), teacher[i], reduction="batchmean") * T * T
            losses.append((1 - alpha) * ce_loss + alpha * kd_loss)
        else:  # kdcl
            kd_loss = F.kl_div(
                F.log_softmax(logits[i] / T, dim=-1), F.softmax(teacher / T, dim=-1), reduction="batchmean"
            ) * T * T
            losses.append((1 - alpha) * ce_loss + alpha * kd_loss)
    return losses


def train_one_epoch(models, optimizers, loader, args):
    for m in models:
        m.train()
    loss_meters = [AverageMeter() for _ in models]
    acc_meters = [AverageMeter() for _ in models]

    for views, labels in loader:
        xs = [views[:, i, :] for i in range(len(models))]
        logits = [model(x) for model, x in zip(models, xs)]
        logits_stack = torch.stack(logits)

        teacher = teacher_logit_for_mode(args.mode, args.ensemble, logits_stack, labels)
        losses = compute_losses(args.mode, args.T, args.alpha, logits, labels, teacher)

        for i, (opt, loss) in enumerate(zip(optimizers, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=(i < len(models) - 1))
            opt.step()
            loss_meters[i].update(loss.item(), n=xs[i].size(0))
            acc = accuracy(logits[i].detach(), labels)[0]
            acc_meters[i].update(acc.item(), n=xs[i].size(0))

    return [m.avg for m in loss_meters], [m.avg for m in acc_meters]


@torch.no_grad()
def evaluate(models, loader):
    for m in models:
        m.eval()
    loss_meters = [AverageMeter() for _ in models]
    acc_meters = [AverageMeter() for _ in models]
    for views, labels in loader:
        for i, model in enumerate(models):
            out = model(views[:, i, :])
            loss = F.cross_entropy(out, labels)
            acc = accuracy(out, labels)[0]
            loss_meters[i].update(loss.item(), n=labels.size(0))
            acc_meters[i].update(acc.item(), n=labels.size(0))
    return [m.avg for m in loss_meters], [m.avg for m in acc_meters]


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    exp_name = f"{args.mode}_{'_'.join(args.model_names)}"
    exp_path = os.path.join("experiments", exp_name, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(exp_path, exist_ok=True)
    log_path = os.path.join(exp_path, "train_log.txt")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"Experiment: {exp_path} (mode={args.mode}, ensemble={args.ensemble})")
    log(f"args: {vars(args)}")

    train_loader, val_loader, in_dim = build_loaders(args)
    models, optimizers, schedulers = [], [], []
    for name in args.model_names:
        model = build_model(name, in_dim, args.n_classes)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        sched = MultiStepLR(opt, args.milestones, args.gamma)
        models.append(model)
        optimizers.append(opt)
        schedulers.append(sched)

    best_acc = [-1.0] * len(BRANCH_NAMES)
    for epoch in range(args.epochs):
        train_losses, train_acces = train_one_epoch(models, optimizers, train_loader, args)
        val_losses, val_acces = evaluate(models, val_loader)
        for i in range(len(BRANCH_NAMES)):
            schedulers[i].step()
            if val_acces[i] > best_acc[i]:
                best_acc[i] = val_acces[i]
                ckpt_dir = os.path.join(exp_path, BRANCH_NAMES[i])
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"epoch": epoch, "model": models[i].state_dict(), "acc": val_acces[i]},
                           os.path.join(ckpt_dir, "best.pth"))

        if (epoch + 1) % args.print_freq == 0:
            for i, name in enumerate(BRANCH_NAMES):
                log(f"[{epoch+1}] {name}: train_loss={train_losses[i]:.3f} train_acc={train_acces[i]:.2f} "
                    f"val_loss={val_losses[i]:.3f} val_acc={val_acces[i]:.2f}")

    for i, name in enumerate(BRANCH_NAMES):
        log(f"{name} best val acc: {best_acc[i]:.2f}")


if __name__ == "__main__":
    main()
