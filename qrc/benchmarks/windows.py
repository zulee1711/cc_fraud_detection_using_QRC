"""Windowing and splitting shared by the benchmark tasks.

Every task turns one input series into overlapping windows, the shape the
backend runs one circuit per, and splits them in time order.
"""

import numpy as np


def sliding_windows(series, window_length):
    """All full windows of `series`, oldest step first.

    Returns:
        array of shape ``(len(series) - window_length + 1, window_length)``;
        row ``i`` is ``series[i : i + window_length]``.
    """
    return np.lib.stride_tricks.sliding_window_view(series, window_length).copy()


def chronological_split(windows, targets, test_split):
    """Reshape windows for the backend and split them in time order.

    Args:
        windows (np.ndarray): shape ``(M, window_length)``, in time order.
        targets (np.ndarray): shape ``(M,)``, one per window.
        test_split (float): fraction of windows, taken from the *end*, held out
            for testing. Windows overlap, so shuffling would leak test inputs
            into training.

    Returns:
        ((X_train, y_train), (X_test, y_test)) with ``X_*`` of shape
        ``(M, window_length, 1)`` — one feature per step — and ``y_*`` of shape ``(M,)``.
    """
    X = windows[:, :, None]
    split_idx = int(len(X) * (1 - test_split))
    return (X[:split_idx], targets[:split_idx]), (X[split_idx:], targets[split_idx:])
