"""NARMA10 benchmark.

The standard stress test for reservoirs (Atiya & Parlos, 2000). The input is
i.i.d. noise ``u_t ~ U[0, 0.5]`` and the target follows the nonlinear
autoregressive moving average recursion

    y_{t+1} = 0.3 y_t + 0.05 y_t (y_t + ... + y_{t-9}) + 1.5 u_{t-9} u_t + 0.1

so predicting it needs both memory (``u_{t-9}``, ten steps back) and
nonlinearity (the products). It is harder than the memory and parity tasks —
the QRC-Lab paper calls it "a deliberate stress test" — so small reservoirs
scoring poorly here is expected, not a bug.

Windowing: the window ending at step ``t`` holds ``u_{t-L+1} .. u_t`` and its
target is ``y_{t+1}``. Each window restarts from |0...0>, so `window_length`
must be at least 10 for ``u_{t-9}`` to be inside it. The ``y`` feedback terms
reach further back than any window, which is part of what makes the task hard.
"""

import numpy as np

from qrc.benchmarks.windows import chronological_split, sliding_windows


class NARMA10Task:
    """
    NARMA10 one-step prediction task.
    Goal: output = y_{t+1} given the inputs up to u_t.
    """

    def __init__(self, length=2000, window_length=10, washout=200, test_split=0.2, seed=None):
        """
        Args:
            length (int): number of time steps in the generated series,
                washout included.
            window_length (int): ``L``, the number of steps encoded per circuit,
                the current one included. Must be at least 10.
            washout (int): initial steps discarded before windowing. The
                recursion starts from ``y = 0``, and these steps let it settle.
            test_split (float): fraction of windows, taken from the *end* of the
                series, held out for testing.
            seed: seed or numpy Generator for the input series.
        """
        if window_length < 10:
            raise ValueError(f"window_length must be at least 10 for NARMA10, got {window_length}")
        if length - washout <= window_length:
            raise ValueError(
                f"length ({length}) minus washout ({washout}) must exceed window_length ({window_length})"
            )

        self.length = length
        self.window_length = window_length
        self.washout = washout
        self.test_split = test_split
        self.seed = seed

    def generate(self):
        """Build the windowed train/test sets.

        Returns:
            ((X_train, y_train), (X_test, y_test)) where ``X_*`` has shape
            ``(M, window_length, 1)`` - oldest steps first - and ``y_*`` has shape ``(M,)``.
        """
        u = np.random.default_rng(self.seed).uniform(0, 0.5, self.length)

        # y[t + 1] is computed from y[t-9 .. t] and u[t], u[t-9].
        y = np.zeros(self.length)
        # A diverging series overflows; that case is reported just below.
        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(9, self.length - 1):
                y[t + 1] = (0.3 * y[t]
                            + 0.05 * y[t] * y[t - 9:t + 1].sum()
                            + 1.5 * u[t - 9] * u[t]
                            + 0.1)
        # NARMA10 is known to blow up for a small fraction of input sequences.
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 10:
            raise RuntimeError(f"NARMA10 series diverged for seed={self.seed!r}; try another seed")

        u, y = u[self.washout:], y[self.washout:]

        # Window t covers u[t-L+1 .. t] and its target is y[t+1], so the last
        # step has no target and is dropped.
        L = self.window_length
        X = sliding_windows(u[:-1], L)  # [M, L]
        target = y[L:]

        # Rescale [0, 0.5] to [0, 1], the range AngleEncoding's default scaling
        # maps to [0, pi]. The recursion above uses the raw u.
        return chronological_split(2 * X, target, self.test_split)
