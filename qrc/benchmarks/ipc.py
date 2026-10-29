"""Information Processing Capacity (IPC) benchmark.

Measures the total computational capabilities of a reservoir across linear and
nonlinear memory degrees using orthogonal Legendre polynomials (Dambre et al.,
2012; Martínez-Peña et al., 2020).

The input is i.i.d. uniform noise s_t ~ U[-1, 1]. The target series are constructed
as products of normalized Legendre polynomials:

    y^{(d)}_t = \prod_j \bar{P}_{d_j}(s_{t - \tau_j})    with    \sum_j d_j = d

Because the polynomial basis is orthonormal over U[-1, 1], the total capacity is
the sum of individual capacities:

    C_total = \sum_d C_d \le N_obs

where N_obs is the number of linearly independent readout observables. A properly
operating reservoir with fading memory saturates this bound (C_total ≈ N_obs).

Windowing:
    Each circuit window ending at step t holds s_{t-L+1} .. s_t. To compute delays
    up to tau_max, window_length L must be at least tau_max + 1. Inputs are mapped
    from [-1, 1] to [0, 1] for AngleEncoding (default scaling maps [0, 1] to [0, pi]).
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import eval_legendre

from qrc.benchmarks.windows import chronological_split, sliding_windows


def _normalized_legendre(d: int, x: np.ndarray) -> np.ndarray:
    """Normalized Legendre polynomial P_d(x) such that E[P_d(x)^2] = 1 on U[-1, 1]."""
    return np.sqrt(2 * d + 1) * eval_legendre(d, x)


class IPCTask:
    """Information Processing Capacity (IPC) benchmark task.

    Evaluates the memory capacity across polynomial degrees (degree 1 = linear memory,
    degree 2+ = nonlinear memory) using orthogonal Legendre polynomials.
    """

    def __init__(
        self,
        length: int = 3000,
        window_length: int = 15,
        max_degree: int = 3,
        max_delay: Optional[int] = None,
        washout: int = 200,
        test_split: float = 0.2,
        threshold: float = 1e-3,
        seed: Optional[int] = None,
    ):
        """
        Args:
            length (int): Total length of the generated input series, washout included.
            window_length (int): Number of steps L encoded per circuit window.
            max_degree (int): Highest polynomial nonlinearity degree (default: 3).
            max_delay (int, optional): Maximum delay tau to probe (tau <= window_length - 1).
                Defaults to window_length - 1.
            washout (int): Initial transient steps discarded before windowing.
            test_split (float): Fraction of windows held out for testing.
            threshold (float): Minimum capacity threshold below which contributions are
                treated as numerical noise and clipped to 0 (default: 1e-3).
            seed: Seed or numpy Generator for reproducibility.
        """
        if max_delay is None:
            max_delay = window_length - 1

        if max_delay >= window_length:
            raise ValueError(
                f"max_delay ({max_delay}) cannot exceed window_length - 1 ({window_length - 1})"
            )
        if length - washout <= window_length:
            raise ValueError(
                f"length ({length}) minus washout ({washout}) must exceed window_length ({window_length})"
            )

        self.length = length
        self.window_length = window_length
        self.max_degree = max_degree
        self.max_delay = max_delay
        self.washout = washout
        self.test_split = test_split
        self.threshold = threshold
        self.seed = seed

        # Populated during generate()
        self.target_degrees: List[int] = []

    def _build_targets(self, s: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Construct orthonormal target columns grouped by total polynomial degree."""
        T = len(s)
        poly_cache = {}
        for d in range(1, self.max_degree + 1):
            for tau in range(self.max_delay + 1):
                shifted = np.roll(s, tau)
                poly_cache[(d, tau)] = _normalized_legendre(d, shifted)

        targets = []
        degrees = []

        # Degree 1: P_1(s_{t - tau})
        for tau in range(self.max_delay + 1):
            targets.append(poly_cache[(1, tau)])
            degrees.append(1)

        # Degree 2: P_2(s_{t - tau}) and P_1(s_{t - tau1}) * P_1(s_{t - tau2})
        if self.max_degree >= 2:
            for tau in range(self.max_delay + 1):
                targets.append(poly_cache[(2, tau)])
                degrees.append(2)
            for tau1, tau2 in itertools.combinations(range(self.max_delay + 1), 2):
                targets.append(poly_cache[(1, tau1)] * poly_cache[(1, tau2)])
                degrees.append(2)

        # Degree 3: P_3, P_2*P_1, P_1*P_1*P_1
        if self.max_degree >= 3:
            for tau in range(self.max_delay + 1):
                targets.append(poly_cache[(3, tau)])
                degrees.append(3)
            for tau1 in range(self.max_delay + 1):
                for tau2 in range(self.max_delay + 1):
                    if tau1 != tau2:
                        targets.append(poly_cache[(2, tau1)] * poly_cache[(1, tau2)])
                        degrees.append(3)
            for tau1, tau2, tau3 in itertools.combinations(
                range(self.max_delay + 1), 3
            ):
                targets.append(
                    poly_cache[(1, tau1)]
                    * poly_cache[(1, tau2)]
                    * poly_cache[(1, tau3)]
                )
                degrees.append(3)

        # [T, num_targets]
        return np.column_stack(targets), degrees

    def generate(self):
        """Build the windowed train/test sets and target matrices.

        Returns:
            ((X_train, Y_train), (X_test, Y_test)) where:
                X_* has shape (M, window_length, 1) mapped to [0, 1] for angle encoding.
                Y_* has shape (M, num_targets) holding orthogonal Legendre targets.
        """
        rng = np.random.default_rng(self.seed)
        # Raw i.i.d. signal in [-1, 1] for orthogonal Legendre polynomials
        s = rng.uniform(-1.0, 1.0, size=self.length)

        # Construct all target combinations across full series
        Y_all, self.target_degrees = self._build_targets(s)

        # Discard washout transient
        s = s[self.washout :]
        Y_all = Y_all[self.washout :]

        # Map s in [-1, 1] to [0, 1] for AngleEncoding
        u = (s + 1.0) / 2.0

        # Window t covers u[t - L + 1 .. t]
        L = self.window_length
        X = sliding_windows(u, L)  # chronological_split adds the feature axis
        Y = Y_all[L - 1 :]  # [M, num_targets]

        return chronological_split(X, Y, self.test_split)

    def compute_capacities(
        self,
        Y_pred: np.ndarray,
        Y_true: np.ndarray,
        n_observables: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute degree-wise and total IPC from readout predictions.

        Args:
            Y_pred (np.ndarray): Predictions from the readout, shape (M, num_targets).
            Y_true (np.ndarray): Target matrix, shape (M, num_targets).

        Returns:
            Dict containing capacity per degree and 'total_capacity'.
        """
        assert (
            Y_pred.shape == Y_true.shape
        ), "Shape mismatch between predictions and targets"

        capacities = {f"degree_{d}": 0.0 for d in range(1, self.max_degree + 1)}
        total_capacity = 0.0

        for col_idx, deg in enumerate(self.target_degrees):
            y_t = Y_true[:, col_idx]
            y_p = Y_pred[:, col_idx]

            var_t = np.var(y_t)
            if var_t < 1e-12:
                continue

            mse = np.mean((y_t - y_p) ** 2)
            c = 1.0 - (mse / var_t)

            # Ignore noise below threshold (Dambre et al., 2012)
            if c > self.threshold:
                capacities[f"degree_{deg}"] += float(c)
                total_capacity += float(c)

        capacities["total_capacity"] = total_capacity

        # Optional normalization by the theoretical bound (Fig. 3c in paper)
        if n_observables is not None and n_observables > 0:
            capacities["normalized_total_capacity"] = total_capacity / n_observables
            for d in range(1, self.max_degree + 1):
                capacities[f"normalized_degree_{d}"] = (
                    capacities[f"degree_{d}"] / n_observables
                )

        return capacities
