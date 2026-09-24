"""Turning the flat feature table into temporal windows.

``DataProcessor.process()`` hands back a 2-D matrix — one row per transaction,
one column per selected feature. The protocol needs a third axis: the backend
runs one circuit per *window*, encoding the window's rows one after another so
the reservoir can carry state between them. This module is the seam between the
two.

Windows are ordered **oldest first**: ``window[0]`` is the earliest transaction
and ``window[-1]`` is the one being classified. That is the order the backend
encodes in, so the reservoir experiences time forwards.
"""

import numpy as np

SPLITS = ("train", "validation", "test")


def window_by_customer_id(input_data, length, splits=SPLITS):
    """Add a timestep axis to the feature matrices in a processor result.

    Args:
        input_data (dict): the return value of ``DataProcessor.process()``. Uses
            ``X_{split}`` and ``groups_{split}`` for every split named in
            `splits`, and passes ``y_{split}`` and ``features`` through
            untouched.
        length (int): window length ``L``, i.e. how many transactions deep each
            window looks — including the one being classified. ``length=1``
            reproduces the old ``X[:, None, :]`` placeholder.
        splits (tuple): split names, **in date order**. The default
            ``("train", "validation", "test")`` is the order the history is
            allowed to reach backwards through.

    Returns:
        dict: the same keys as `input_data`, with each ``X_{split}`` promoted
        from ``(n, features)`` to ``(n, length, features)``, plus a boolean
        ``has_full_history_{split}`` of shape ``(n,)``. ``y_{split}``,
        ``groups_{split}`` and ``features`` are unchanged — row ``i`` of every
        array still describes the same transaction.
    """
    if length < 1:
        raise ValueError(f"length must be at least 1, got {length}")

    matrices, keys, lengths = [], [], []
    for split in splits:
        X = np.asarray(input_data[f"X_{split}"])
        group = np.asarray(input_data[f"groups_{split}"])
        y = np.asarray(input_data[f"y_{split}"])

        if X.ndim != 2:
            raise ValueError(
                f"X_{split} must have shape (samples, features), got {X.shape}"
            )
        if len(group) != len(X) or len(y) != len(X):
            raise ValueError(
                f"{split}: X ({len(X)}), groups ({len(group)}) and y ({len(y)}) "
                "must describe the same rows"
            )

        matrices.append(X)
        keys.append(group)
        lengths.append(len(X))

    if len({X.shape[1] for X in matrices}) != 1:
        raise ValueError(
            "every split must carry the same features: "
            + ", ".join(f"{s}={X.shape[1]}" for s, X in zip(splits, matrices))
        )

    # One pass over the whole timeline, so history flows across the boundaries.
    windows, has_full_history = _window(
        np.concatenate(matrices),
        np.concatenate(keys),
        length,
    )

    windowed = {"features": input_data["features"]}
    start = 0
    for split, group, size in zip(splits, keys, lengths):
        stop = start + size
        windowed[f"X_{split}"] = windows[start:stop]
        windowed[f"y_{split}"] = np.asarray(input_data[f"y_{split}"])
        windowed[f"groups_{split}"] = group
        windowed[f"has_full_history_{split}"] = has_full_history[start:stop]
        start = stop

    return windowed


def _window(X, groups, length):
    """Build one window per row of `X`, gathering backwards within each group.

    Args:
        X (ndarray): shape ``(rows, features)``, sorted by time.
        groups (ndarray): shape ``(rows,)``, the customer of each row.
        length (int): window length.

    Returns:
        tuple: ``(windows, has_full_history)`` with shapes
        ``(rows, length, features)`` and ``(rows,)``.
    """
    rows, features = X.shape
    windows = np.empty((rows, length, features), dtype=X.dtype)
    has_full_history = np.empty(rows, dtype=bool)

    order = np.argsort(groups, kind="stable")
    ordered = groups[order]
    starts = np.flatnonzero(
        np.concatenate(([True], ordered[1:] != ordered[:-1], [True]))
    )

    for start, stop in zip(starts[:-1], starts[1:]):
        rows_of_customer = order[start:stop]
        position = np.arange(stop - start)

        for step in range(length):
            # step 0 is the oldest slot, step length-1 the transaction itself.
            back = np.maximum(position - (length - 1 - step), 0)
            windows[rows_of_customer, step] = X[rows_of_customer[back]]

        has_full_history[rows_of_customer] = position >= length - 1

    return windows, has_full_history
