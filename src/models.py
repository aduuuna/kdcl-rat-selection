"""Per-branch tabular encoders (replaces ResNet/VGG/WRN/... from the reference repo). All
architectures share the same output dimension so their logits are ensemble-compatible."""
import torch.nn as nn
import torch.nn.functional as F


class RATMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GateNet(nn.Module):
    """[mode=one] Small gating network: takes the raw (averaged) input feature vector and produces
    one softmax weight per branch, used to combine branch logits into a learned ensemble teacher.
    This is an ADAPTATION of ONE (Lan et al. 2018) -- the canonical architecture shares low-level
    trunk layers across branches with a gating head on top; our two branches (rat_mlp_small /
    rat_mlp_large) are fully separate architectures with no shared trunk, so this external gate
    reading the input directly stands in for that. Document this substitution explicitly in the
    write-up -- it's a real deviation, not the textbook method."""
    def __init__(self, in_dim: int, n_branches: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_branches),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


def build_model(name: str, in_dim: int, n_classes: int) -> nn.Module:
    if name not in model_dict:
        raise ValueError(f"Unknown model name '{name}', options: {list(model_dict)}")
    return model_dict[name](in_dim=in_dim, n_classes=n_classes)


model_dict = {
    "rat_mlp_tiny": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=()),
    "rat_mlp_small": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=(64, 32)),
    "rat_mlp": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=(128, 64)),
    "rat_mlp_large": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=(256, 128, 64)),
}