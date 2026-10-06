import numpy as np
import pytest

from experiments.lstm_baseline import run_lstm_baseline


def test_lstm_baseline_trains_and_reports_classification_metrics():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)

    def make_split(size):
        X = rng.normal(size=(size, 3, 2)).astype(np.float32)
        y = (X[:, -1, 0] > 0).astype(int)
        return X, y

    results = run_lstm_baseline(
        {
            "train": make_split(64),
            "validation": make_split(24),
            "test": make_split(24),
        },
        seed=0,
        hidden_size=4,
        epochs=3,
        batch_size=16,
        patience=2,
    )

    assert results["backend"] == "lstm"
    assert results["seconds"] >= 0
    assert set(results["metrics"]) == {"validation", "test"}
    assert set(results["scores"]) == {"validation", "test"}
    for split in ("validation", "test"):
        assert len(results["scores"][split]) == 24
        assert results["metrics"][split]["threshold_used"] == 0.5
        for metric in ("pr_auc", "f1_score", "precision", "recall", "roc_auc"):
            assert np.isfinite(results["metrics"][split][metric])