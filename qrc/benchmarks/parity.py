"""Temporal parity (XOR) benchmark.

The reservoir is fed a stream of random bits and the readout must output
the parity (XOR) of the last `order` of them. Storing the bits is not enough
here — for ``order >= 2`` the target is not a linear function of the inputs,
so a linear readout can only solve it if the reservoir has already mixed the
bits together nonlinearly.

``order=1`` is just "recall the current bit", i.e. the memory task at delay 0
with binary inputs — a useful baseline for the higher orders.

As for the memory task, each window restarts from |0...0>, so the bits being
XORed must all fit inside the window: `order` is at most `window_length`.
"""

import numpy as np

from qrc.benchmarks.windows import chronological_split, sliding_windows


class ParityTask:
    """
    Temporal parity task.
    Goal: output = input_t XOR input_{t-1} XOR ... XOR input_{t-order+1}
    """

    def __init__(self, order=2, length=1000, window_length=10, test_split=0.2, seed=None):
        """
        Args:
            order (int): how many of the most recent bits are XORed together,
                the current one included.
            length (int): number of time steps in the generated series.
            window_length (int): ``L``, the number of steps encoded per circuit,
                the current one included. Must be at least `order`.
            test_split (float): fraction of windows, taken from the *end* of the
                series, held out for testing.
            seed: seed or numpy Generator for the input bits. Keep it fixed
                across an order sweep so every order is scored on the same data.
        """
        if not 1 <= order <= window_length:
            raise ValueError(
                f"order must satisfy 1 <= order <= window_length, got order={order}, "
                f"window_length={window_length}"
            )
        if length < window_length:
            raise ValueError(f"length ({length}) must be at least window_length ({window_length})")

        self.order = order
        self.length = length
        self.window_length = window_length
        self.test_split = test_split
        self.seed = seed

    def generate(self):
        """Build the windowed train/test sets.

        Returns:
            ((X_train, y_train), (X_test, y_test)) where ``X_*`` has shape
            ``(M, window_length, 1)`` - oldest steps first - and ``y_*`` has
            shape ``(M,)`` with values in {0, 1}.
        """
        # Random bits: AngleEncoding's default scaling maps them to Ry(0) or Ry(pi).
        u = np.random.default_rng(self.seed).integers(0, 2, self.length).astype(float)

        # Window t covers u[t-L+1 .. t]; its target is the parity of its last `order` steps.
        L = self.window_length
        X = sliding_windows(u, L)  # [M, L]
        y = X[:, L - self.order:].sum(axis=1) % 2

        return chronological_split(X, y, self.test_split)
