"""Channel-impairment distortions for tabular state vectors -- the ICL step, replacing image
crop/flip augmentation."""
import numpy as np


class AWGNDistortion:
    """Additive white Gaussian noise on the feature vector."""

    def __init__(self, sigma: float = 0.05, seed: int = 0):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.sigma, size=x.shape).astype(np.float32)
        return x + noise


class RayleighFadingDistortion:
    """Multiplicative Rayleigh-fading-like scaling per feature."""

    def __init__(self, scale: float = 0.1, seed: int = 0):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        fade = self.rng.rayleigh(scale=self.scale, size=x.shape).astype(np.float32)
        return x * (1.0 - fade)


class DelayJitterDistortion:
    """Simulates a delayed/stale reading by randomly zeroing a fraction of features."""

    def __init__(self, drop_prob: float = 0.05, seed: int = 0):
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mask = self.rng.random(size=x.shape) > self.drop_prob
        return x * mask.astype(np.float32)


class NoDistortion:
    """Identity -- used for the ICL ablation (both branches see the clean, undistorted row)."""

    def __init__(self, seed: int = 0):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


DISTORTION_REGISTRY = {
    "awgn": AWGNDistortion,
    "fading": RayleighFadingDistortion,
    "delay": DelayJitterDistortion,
    "none": NoDistortion,
}
