"""Classical fixed random feature maps: tanh and RFF."""

from abc import ABC, abstractmethod

import numpy as np


class ClassicalRandomMap(ABC):
    """Base class for fixed classical random feature maps."""

    name = "base"

    def __init__(self, input_width, map_width=None, scale=1.0, seed=42):
        if input_width < 1:
            raise ValueError("input_width must be positive")
        if map_width is not None and map_width < 1:
            raise ValueError("map_width must be positive")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")

        self.input_width = input_width
        self.map_width = input_width if map_width is None else map_width
        self.scale = scale
        self.seed = seed
        self.weights = None
        self.bias = None

    def fit(self, X=None):
        """Initialise fixed random parameters; no labels are used."""
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2 or X.shape[1] != self.input_width:
                raise ValueError("X has an unexpected shape")

        rng = np.random.default_rng(self.seed)
        self._initialise_parameters(rng)
        return self

    @abstractmethod
    def _initialise_parameters(self, rng):
        pass

    @abstractmethod
    def _apply(self, Z):
        pass

    def transform(self, X):
        """Transform a 2D input feature matrix."""
        if self.weights is None:
            raise RuntimeError("Call fit() before transform()")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.input_width:
            raise ValueError(f"Expected X with shape (n_samples, {self.input_width})")

        return self._apply(X @ self.weights + self.bias)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_metadata(self):
        return {
            "classical_model": self.name,
            "random_map_width": self.map_width,
            "random_map_scale": self.scale,
            "random_map_seed": self.seed,
        }


class TanhRandomMap(ClassicalRandomMap):
    """Fixed random projection followed by tanh."""

    name = "tanh"

    def _initialise_parameters(self, rng):
        self.weights = rng.normal(
            loc=0.0,
            scale=self.scale / np.sqrt(self.input_width),
            size=(self.input_width, self.map_width),
        )
        self.bias = rng.uniform(-1.0, 1.0, size=self.map_width)

    def _apply(self, Z):
        return np.tanh(Z)


class RandomFourierFeatures(ClassicalRandomMap):
    """Random Fourier features approximating an RBF kernel."""

    name = "rff"

    def __init__(
        self,
        input_width,
        map_width=None,
        scale=1.0,
        seed=42,
        gamma=1.0,
    ):
        super().__init__(input_width, map_width, scale, seed)

        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")

        self.gamma = gamma

    def _initialise_parameters(self, rng):
        # For the RBF kernel exp(-gamma * ||x-y||^2),
        # frequencies are sampled from N(0, 2*gamma).
        self.weights = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 * self.gamma),
            size=(self.input_width, self.map_width),
        )
        self.bias = rng.uniform(0.0, 2.0 * np.pi, size=self.map_width)

    def _apply(self, Z):
        return np.sqrt(2.0 / self.map_width) * np.cos(Z)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["rff_gamma"] = self.gamma
        return metadata


SUPPORTED_RANDOM_MAPS = {
    "tanh": TanhRandomMap,
    "rff": RandomFourierFeatures,
}


def make_random_map(method, **kwargs):
    """Construct a random map by name."""
    try:
        map_class = SUPPORTED_RANDOM_MAPS[method]
    except KeyError:
        raise ValueError(
            f"Unknown random map {method!r}. "
            f"Choose from {tuple(SUPPORTED_RANDOM_MAPS)}."
        ) from None

    return map_class(**kwargs)
