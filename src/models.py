"""Per-branch tabular encoders (replaces ResNet/VGG/WRN/... from the reference repo). All
architectures share the same output dimension so their logits are ensemble-compatible."""
import torch.nn as nn


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


def build_model(name: str, in_dim: int, n_classes: int) -> nn.Module:
    if name not in model_dict:
        raise ValueError(f"Unknown model name '{name}', options: {list(model_dict)}")
    return model_dict[name](in_dim=in_dim, n_classes=n_classes)


model_dict = {
    "rat_mlp_tiny": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=()),  # pure linear, no hidden layer
    "rat_mlp_small": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=(64, 32)),
    "rat_mlp": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=(128, 64)),
    "rat_mlp_large": lambda in_dim, n_classes: RATMLP(in_dim, n_classes, hidden=(256, 128, 64)),
}
