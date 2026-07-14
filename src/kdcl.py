"""KDCL teacher-logit ensemble methods (paper Sec. 3). `naive`/`minlogit`/`linear` implement the
actual paper math -- unlike reference/train.py, which just averages logits."""
import torch
import torch.nn.functional as F


def naive(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (m, batch, n_classes) -> per-sample logits of whichever student has lowest CE loss."""
    m, batch, _ = logits.shape
    ce_per_student = torch.stack([
        F.cross_entropy(logits[i], labels, reduction="none") for i in range(m)
    ])
    best = ce_per_student.argmin(dim=0)
    return logits[best, torch.arange(batch)]


def minlogit(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Normalize each student's logits against its own target-class logit, then take the
    element-wise minimum across students -- guaranteed no worse than any single student."""
    m, batch, n_classes = logits.shape
    target_logit = logits.gather(2, labels.view(1, batch, 1).expand(m, batch, 1))
    normalized = logits - target_logit
    teacher_logit, _ = normalized.min(dim=0)
    return teacher_logit


def linear(logits: torch.Tensor, labels: torch.Tensor, n_iters: int = 50, lr: float = 0.1) -> torch.Tensor:
    """Convex combination of student logits minimizing training CE loss (paper Eq. 5), solved
    via projected gradient descent rather than a full QP solver."""
    m = logits.shape[0]
    alpha = torch.full((m,), 1.0 / m, device=logits.device, requires_grad=True)
    for _ in range(n_iters):
        combined = torch.einsum("m,mbc->bc", alpha, logits)
        loss = F.cross_entropy(combined, labels)
        grad, = torch.autograd.grad(loss, alpha)
        with torch.no_grad():
            alpha -= lr * grad
            alpha.clamp_(min=0)
            alpha /= alpha.sum().clamp(min=1e-8)
    with torch.no_grad():
        teacher_logit = torch.einsum("m,mbc->bc", alpha, logits)
    return teacher_logit


ENSEMBLE_METHODS = {
    "naive": naive,
    "minlogit": minlogit,
    "linear": linear,
}
