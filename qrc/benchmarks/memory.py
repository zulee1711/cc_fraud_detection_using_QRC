"""Short-term memory benchmark.

A sanity check for the QRC pipeline before it sees fraud data: the reservoir is
fed a stream of i.i.d. random numbers and the readout must reproduce the input
from `delay` steps ago. The answer is known exactly, so a bad score points at
the reservoir (or the pipeline), not at the data.

The backend runs one circuit per *window* and restarts from |0...0> each time,
so the reservoir can only ever remember what is inside the window: delays must
be strictly smaller than `window_length`.
"""

import numpy as np

from qrc.benchmarks.windows import chronological_split, sliding_windows


class MemoryTask:
    """
    Short-term memory task.
    Goal: output = input_{t-delay}
    """

    def __init__(self, delay=1, length=1000, window_length=10, test_split=0.2, seed=None):
        """
        Args:
            delay (int): how many steps back the target looks. ``0`` just asks
                for the current input.
            length (int): number of time steps in the generated series.
            window_length (int): ``L``, the number of steps encoded per circuit,
                the current one included. Must be larger than `delay`.
            test_split (float): fraction of windows, taken from the *end* of the
                series, held out for testing.
            seed: seed or numpy Generator for the input series. Keep it fixed
                across a delay sweep so every delay is scored on the same data.
        """
        if not 0 <= delay < window_length:
            raise ValueError(
                f"delay must satisfy 0 <= delay < window_length, got delay={delay}, "
                f"window_length={window_length}"
            )
        if length < window_length:
            raise ValueError(f"length ({length}) must be at least window_length ({window_length})")

        self.delay = delay
        self.length = length
        self.window_length = window_length
        self.test_split = test_split
        self.seed = seed

    def generate(self):
        """Build the windowed train/test sets.

        Returns:
            ((X_train, y_train), (X_test, y_test)) where ``X_*`` has shape
            ``(M, window_length, 1)`` - oldest steps first - and ``y_*`` has shape ``(M,)``.
        """
        # Random inputs in [0, 1]: AngleEncoding's default scaling maps them to [0, pi].
        u = np.random.default_rng(self.seed).uniform(0, 1, self.length)

        # Window t covers u[t-L+1 .. t] and its target is u[t - delay]. Only full
        # windows are kept, so every target is a real input (no zero padding).
        L = self.window_length
        X = sliding_windows(u, L)  # [M, L]
        y = X[:, L - 1 - self.delay]

        return chronological_split(X, y, self.test_split)
