"""Multi-branch training loop for the 3G/4G RAT students on the RTR-NetTest dataset.

Modes:
  vanilla     -- each branch trained alone, no collaboration (control group).
  dml         -- Deep Mutual Learning: each branch's KD target is the *other* branch's raw
                 softmax probabilities, no protection against a weak branch dragging a strong one
                 down.
  kdcl        -- this project's core method. --ensemble minlogit is the paper's protected
                 ensembling; --ensemble linear is KDCL-Linear (already implemented in kdcl.py,
                 never run on this dataset before -- just pass --ensemble linear).
  one         -- ADAPTED On-the-fly Native Ensemble (Lan et al., 2018). See GateNet's docstring in
                 models.py for exactly what's adapted vs. the canonical shared-trunk version. A
                 small external GateNet reads the (averaged) input features, produces a softmax
                 weight per branch, and combines the branches' (detached) logits into a learned
                 ensemble teacher; each branch is then KD-trained against that teacher (same
                 mechanics as kdcl/dml), and the gate is trained separately with its own CE loss.
  offline_kd  -- classic, asymmetric, OFFLINE teacher-student KD. Requires --teacher_ckpt pointing
                 at an already-trained experiment dir (typically a vanilla run). The branch at
                 index 1 (rat_mlp_large / 4G_agent by default) is loaded from that checkpoint,
                 frozen, and used as a fixed teacher for a fresh student (index 0 / rat_mlp_small /
                 3G_agent). Only the student is trained/saved in this mode.
"""
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from rat_dataset import (
    load_raw, make_load_label, numeric_feature_columns,
    time_blocked_split, N_LOAD_CLASSES, LABEL_COL, LOAD_COL, fit_scaler, RATS,
)
from distortions import DISTORTION_REGISTRY
from kdcl import ENSEMBLE_METHODS
from models import build_model, GateNet
from utils import AverageMeter, accuracy

BRANCH_NAMES = (f"{RATS[0]}_agent", f"{RATS[1]}_agent")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "dml", "kdcl", "one", "offline_kd"], default="kdcl")
    parser.add_argument("--ensemble", choices=list(ENSEMBLE_METHODS), default="minlogit",
                         help="only used when --mode kdcl. 'linear' = KDCL-Linear.")
    parser.add_argument("--model_names", type=str, nargs=2, default=["rat_mlp_small", "rat_mlp_large"],
                         help=f"one model name per branch, order {BRANCH_NAMES[0]} {BRANCH_NAMES[1]}. "
                              f"For --mode offline_kd: index 0 = student arch, index 1 = the arch "
                              f"the frozen teacher checkpoint was trained with.")
    parser.add_argument("--distortions", type=str, nargs=2, default=["awgn", "fading"],
                         help=f"one distortion per branch, order {BRANCH_NAMES[0]} {BRANCH_NAMES[1]}")
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
    parser.add_argument("--n_classes", type=int, default=N_LOAD_CLASSES)
    parser.add_argument("--include_tech_flag", action="store_true",
                         help=f"keep 'is_{RATS[1]}' as a feature -- legacy/comparison mode only.")
    parser.add_argument("--gate_hidden", type=int, default=32, help="[mode=one] GateNet hidden width")
    parser.add_argument("--gate_lr", type=float, default=1e-3, help="[mode=one] GateNet learning rate")
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                         help=f"[mode=offline_kd, required] path to an experiment dir containing "
                              f"'{BRANCH_NAMES[1]}/best.pth' to use as the frozen teacher")
    return parser.parse_args()


class FastMultiViewLoader:
    """Drop-in replacement for DataLoader(MultiViewDataset(...)) that applies each branch's
    distortion once per BATCH (numpy broadcasting) instead of once per SAMPLE via __getitem__.
    Confirmed 2026-09-23 (200K-row 4G/5G dataset): per-sample __getitem__ overhead dominates epoch
    time, not matmul cost -- raising batch_size 64->512 under the old DataLoader path only cut
    epoch time ~2x instead of the ~8x expected from fewer optimizer steps alone. Yields the same
    (views, labels) shape the old MultiViewDataset did, so train_one_epoch/evaluate are unchanged.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, distortions: list,
                 batch_size: int, shuffle: bool, seed: int = 0):
        self.features = features
        self.labels = labels
        self.distortions = distortions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.labels)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            base = self.features[batch_idx]  # (b, F)
            views = np.stack([d(base.copy()) for d in self.distortions])  # (n_branches, b, F)
            views = views.transpose(1, 0, 2)  # (b, n_branches, F) -- matches views[:, i, :] usage
            yield torch.from_numpy(views), torch.from_numpy(self.labels[batch_idx])

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


def _scaled_features(df, feature_cols, scaler):
    mean, std = scaler
    values = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    return (values - mean) / std


def build_loaders(args):
    df = load_raw()
    feature_cols = numeric_feature_columns(df, include_tech_flag=args.include_tech_flag)
    df[LABEL_COL] = make_load_label(df, load_col=LOAD_COL, n_classes=args.n_classes)
    train_df, val_df, _ = time_blocked_split(df, order_cols=("time_utc",))
    scaler = fit_scaler(train_df, feature_cols)

    distortions = [DISTORTION_REGISTRY[args.distortions[i]](seed=args.seed + i) for i in range(2)]
    train_features = _scaled_features(train_df, feature_cols, scaler)
    train_labels = train_df[LABEL_COL].to_numpy(dtype=np.int64)
    val_features = _scaled_features(val_df, feature_cols, scaler)
    val_labels = val_df[LABEL_COL].to_numpy(dtype=np.int64)

    train_loader = FastMultiViewLoader(train_features, train_labels, distortions,
                                        batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = FastMultiViewLoader(val_features, val_labels, [lambda x: x, lambda x: x],
                                      batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, len(feature_cols)


def teacher_logit_for_mode(mode, ensemble_name, logits_stack, labels):
    """Detached teacher logit for dml/kdcl (compute_losses treats kdcl/one identically)."""
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
        else:  # kdcl or one -- raw-logit teacher, softened by temperature
            kd_loss = F.kl_div(
                F.log_softmax(logits[i] / T, dim=-1), F.softmax(teacher / T, dim=-1), reduction="batchmean"
            ) * T * T
            losses.append((1 - alpha) * ce_loss + alpha * kd_loss)
    return losses


def train_one_epoch(models, optimizers, loader, args, gate=None, gate_optimizer=None):
    for m in models:
        m.train()
    if gate is not None:
        gate.train()
    loss_meters = [AverageMeter() for _ in models]
    acc_meters = [AverageMeter() for _ in models]
    gate_acc_meter = AverageMeter() if gate is not None else None

    for views, labels in loader:
        xs = [views[:, i, :] for i in range(len(models))]
        logits = [model(x) for model, x in zip(models, xs)]
        logits_stack = torch.stack(logits)

        if args.mode == "one":
            gate_input = sum(xs) / len(xs)  # average of the two distorted views -- cheap proxy
                                             # for "what the row looked like" (no clean view here)
            weights = gate(gate_input)                       # (batch, n_branches), differentiable
            detached_stack = logits_stack.detach()
            teacher = torch.einsum("bm,mbc->bc", weights.detach(), detached_stack)  # for KD
            losses = compute_losses(args.mode, args.T, args.alpha, logits, labels, teacher)

            gate_ensemble = torch.einsum("bm,mbc->bc", weights, detached_stack)  # live, for gate only
            gate_loss = F.cross_entropy(gate_ensemble, labels)
            gate_optimizer.zero_grad()
            gate_loss.backward()
            gate_optimizer.step()
            gate_acc_meter.update(accuracy(gate_ensemble.detach(), labels)[0].item(), n=labels.size(0))
        else:
            teacher = teacher_logit_for_mode(args.mode, args.ensemble, logits_stack, labels)
            losses = compute_losses(args.mode, args.T, args.alpha, logits, labels, teacher)

        for i, (opt, loss) in enumerate(zip(optimizers, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=(i < len(models) - 1))
            opt.step()
            loss_meters[i].update(loss.item(), n=xs[i].size(0))
            acc = accuracy(logits[i].detach(), labels)[0]
            acc_meters[i].update(acc.item(), n=xs[i].size(0))

    if gate is not None:
        return [m.avg for m in loss_meters], [m.avg for m in acc_meters], gate_acc_meter.avg
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


def train_one_epoch_offline_kd(student, teacher, optimizer, loader, args):
    """Classic offline KD: the same (single, distorted) input feeds both the frozen teacher and
    the trainable student -- unlike the collaborative modes above, the teacher never updates."""
    student.train()
    teacher.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for views, labels in loader:
        x = views[:, 0, :]
        student_logits = student(x)
        with torch.no_grad():
            teacher_logits = teacher(x)
        ce_loss = F.cross_entropy(student_logits, labels)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / args.T, dim=-1),
            F.softmax(teacher_logits / args.T, dim=-1),
            reduction="batchmean",
        ) * args.T * args.T
        loss = (1 - args.alpha) * ce_loss + args.alpha * kd_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy(student_logits.detach(), labels)[0].item(), n=x.size(0))
    return loss_meter.avg, acc_meter.avg


def run_offline_kd(args, exp_path, log):
    if args.teacher_ckpt is None:
        raise ValueError("--mode offline_kd requires --teacher_ckpt <experiment dir>, e.g. "
                          "experiments/netztest_vanilla/2026-07-28-14-27-00")
    train_loader, val_loader, in_dim = build_loaders(args)

    student = build_model(args.model_names[0], in_dim, args.n_classes)
    teacher = build_model(args.model_names[1], in_dim, args.n_classes)
    teacher_state = torch.load(
        os.path.join(args.teacher_ckpt, BRANCH_NAMES[1], "best.pth"), map_location="cpu"
    )
    teacher.load_state_dict(teacher_state["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log(f"Loaded frozen teacher ({args.model_names[1]}) from {args.teacher_ckpt}, "
        f"val_acc={teacher_state.get('acc', 'n/a')}")

    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)

    best_acc = -1.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch_offline_kd(student, teacher, optimizer, train_loader, args)
        (val_loss,), (val_acc,) = evaluate([student], val_loader)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = os.path.join(exp_path, BRANCH_NAMES[0])
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"epoch": epoch, "model": student.state_dict(), "acc": val_acc},
                       os.path.join(ckpt_dir, "best.pth"))
        if (epoch + 1) % args.print_freq == 0:
            log(f"[{epoch+1}] {BRANCH_NAMES[0]} (student): train_loss={train_loss:.3f} "
                f"train_acc={train_acc:.2f} val_loss={val_loss:.3f} val_acc={val_acc:.2f}")
    log(f"{BRANCH_NAMES[0]} (student, offline_kd) best val acc: {best_acc:.2f}")


def run_collaborative(args, exp_path, log):
    """vanilla / dml / kdcl / one -- both branches trained together each epoch."""
    train_loader, val_loader, in_dim = build_loaders(args)
    models, optimizers, schedulers = [], [], []
    for name in args.model_names:
        model = build_model(name, in_dim, args.n_classes)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        sched = MultiStepLR(opt, args.milestones, args.gamma)
        models.append(model)
        optimizers.append(opt)
        schedulers.append(sched)

    gate, gate_optimizer = None, None
    if args.mode == "one":
        gate = GateNet(in_dim, n_branches=len(models), hidden=args.gate_hidden)
        gate_optimizer = optim.Adam(gate.parameters(), lr=args.gate_lr)

    best_acc = [-1.0] * len(BRANCH_NAMES)
    for epoch in range(args.epochs):
        if args.mode == "one":
            train_losses, train_acces, gate_acc = train_one_epoch(
                models, optimizers, train_loader, args, gate=gate, gate_optimizer=gate_optimizer
            )
        else:
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
            if args.mode == "one":
                log(f"[{epoch+1}] gate: train_acc={gate_acc:.2f}")

    for i, name in enumerate(BRANCH_NAMES):
        log(f"{name} best val acc: {best_acc[i]:.2f}")


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    suffix = "_withtechflag" if args.include_tech_flag else ""
    exp_name = f"netztest_{args.mode}"
    if args.mode == "kdcl" and args.ensemble != "minlogit":
        exp_name += f"_{args.ensemble}"
    exp_name += suffix
    # seconds (not just minutes) in the timestamp -- avoids collisions when a seed-sweep script
    # launches several runs back to back within the same minute.
    exp_path = os.path.join("experiments", exp_name, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(exp_path, exist_ok=True)
    log_path = os.path.join(exp_path, "train_log.txt")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"Experiment: {exp_path} (mode={args.mode}, ensemble={args.ensemble}, seed={args.seed})")
    log(f"args: {vars(args)}")

    if args.mode == "offline_kd":
        run_offline_kd(args, exp_path, log)
    else:
        run_collaborative(args, exp_path, log)


if __name__ == "__main__":
    main()