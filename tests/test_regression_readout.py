import numpy as np
import pytest

from qrc.readout import RegressionReadout


def test_linear_target_is_recovered_exactly():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0

    metrics, y_pred = RegressionReadout().fit_evaluate(X[:150], y[:150], X[150:], y[150:])

    assert y_pred.shape == (50,)
    assert metrics["nmse"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["capacity"] == pytest.approx(1.0)


def test_uninformative_features_give_zero_capacity():
    """Constant features give a constant prediction: capacity 0, not NaN."""
    X = np.ones((100, 3))
    y = np.random.default_rng(0).normal(size=100)

    metrics, _ = RegressionReadout().fit_evaluate(X[:80], y[:80], X[80:], y[80:])

    assert metrics["capacity"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["nmse"] >= 1.0
