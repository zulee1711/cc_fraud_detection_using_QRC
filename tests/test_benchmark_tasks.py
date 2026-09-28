"""The benchmark tasks build the targets they claim to.

No reservoir here: these check the data the benchmarks score against, so a bad
benchmark score can be blamed on the reservoir and not on the task.
"""

import numpy as np
import pytest

from qrc.benchmarks import MemoryTask, NARMA10Task, ParityTask
from qrc.benchmarks.windows import chronological_split, sliding_windows


def _all(splits):
    """Concatenate ((X_train, y_train), (X_test, y_test)) back into (X, y)."""
    (X_train, y_train), (X_test, y_test) = splits
    return np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])


def test_sliding_windows_rows_are_consecutive_slices():
    series = np.arange(6.0)
    windows = sliding_windows(series, 3)
    assert windows.shape == (4, 3)
    for i, row in enumerate(windows):
        np.testing.assert_array_equal(row, series[i:i + 3])


def test_chronological_split_keeps_time_order():
    windows = np.arange(20.0).reshape(10, 2)
    (X_train, y_train), (X_test, y_test) = chronological_split(windows, np.arange(10.0), test_split=0.3)

    assert X_train.shape == (7, 2, 1) and X_test.shape == (3, 2, 1)
    np.testing.assert_array_equal(y_train, np.arange(7.0))
    np.testing.assert_array_equal(y_test, np.arange(7.0, 10.0))


@pytest.mark.parametrize("delay", [0, 1, 4])
def test_memory_target_is_the_delayed_input(delay):
    L = 5
    X, y = _all(MemoryTask(delay=delay, length=50, window_length=L, seed=0).generate())

    assert X.shape == (50 - L + 1, L, 1)
    np.testing.assert_array_equal(y, X[:, L - 1 - delay, 0])


@pytest.mark.parametrize("order", [1, 2, 5])
def test_parity_target_is_the_xor_of_the_last_bits(order):
    L = 5
    X, y = _all(ParityTask(order=order, length=50, window_length=L, seed=0).generate())

    bits = X[:, :, 0].astype(int)
    assert set(np.unique(bits)) <= {0, 1}
    expected = np.bitwise_xor.reduce(bits[:, L - order:], axis=1)
    np.testing.assert_array_equal(y, expected)


def test_narma_target_follows_the_recursion():
    """Rebuild y_{t+1} from the window's inputs and the previous targets."""
    L, washout = 10, 50
    X, y = _all(NARMA10Task(length=300, window_length=L, washout=washout, seed=0).generate())

    u_raw = np.random.default_rng(0).uniform(0, 0.5, 300)
    np.testing.assert_allclose(X[:, :, 0], 2 * sliding_windows(u_raw[washout:-1], L))

    # Window i ends at u_t and targets y_{t+1}; y_t .. y_{t-9} are the previous
    # ten targets, so the check starts once those are all available.
    u = X[:, :, 0] / 2
    for i in range(10, len(y)):
        y_prev = y[i - 10:i]
        expected = 0.3 * y_prev[-1] + 0.05 * y_prev[-1] * y_prev.sum() + 1.5 * u[i, 0] * u[i, -1] + 0.1
        assert y[i] == pytest.approx(expected)


def test_tasks_are_reproducible_for_a_fixed_seed():
    for make in (lambda: MemoryTask(delay=2, length=50, window_length=5, seed=7),
                 lambda: ParityTask(order=2, length=50, window_length=5, seed=7),
                 lambda: NARMA10Task(length=300, washout=50, seed=7)):
        X_a, y_a = _all(make().generate())
        X_b, y_b = _all(make().generate())
        np.testing.assert_array_equal(X_a, X_b)
        np.testing.assert_array_equal(y_a, y_b)


@pytest.mark.parametrize("make", [
    lambda: MemoryTask(delay=5, window_length=5),
    lambda: MemoryTask(delay=-1),
    lambda: MemoryTask(length=3, window_length=5),
    lambda: ParityTask(order=0),
    lambda: ParityTask(order=6, window_length=5),
    lambda: NARMA10Task(window_length=9),
    lambda: NARMA10Task(length=205, washout=200),
])
def test_invalid_settings_are_rejected(make):
    with pytest.raises(ValueError):
        make()
