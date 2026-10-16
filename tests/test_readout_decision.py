import numpy as np
import pytest

from qrc.readout import ClassicalReadout


def test_readout_selects_and_stores_validation_threshold(monkeypatch):
    readout = ClassicalReadout()
    scores = np.array([0.9, 0.7, 0.6, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0, 0])

    monkeypatch.setattr(readout, "predict", lambda _: scores)

    threshold = readout.select_decision_threshold(
        np.zeros((len(labels), 1)),
        labels,
        min_recall=0.8,
    )

    assert threshold == pytest.approx(0.6)
    assert readout.decision_threshold == pytest.approx(0.6)


def test_evaluate_reuses_selected_threshold_and_reports_fpr(monkeypatch):
    readout = ClassicalReadout(decision_threshold=0.7)
    scores = np.array([0.9, 0.8, 0.65, 0.2])
    labels = np.array([1, 0, 1, 0])

    monkeypatch.setattr(readout, "predict", lambda _: scores)

    metrics, _, predictions, returned_scores = readout.evaluate(
        np.zeros((len(labels), 1)),
        labels,
    )

    np.testing.assert_array_equal(predictions, [1, 1, 0, 0])
    np.testing.assert_array_equal(returned_scores, scores)
    assert metrics["threshold_used"] == pytest.approx(0.7)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)