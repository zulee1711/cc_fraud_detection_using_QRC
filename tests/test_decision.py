#%%
import numpy as np
import pytest

from qrc.decision import card_precision_top_k, select_threshold_min_fp

#%%
def test_threshold_minimizes_false_positives_subject_to_recall():
    # To keep recall >= 0.8 with two frauds, both frauds must be detected.
    # Threshold 0.6 does that with one false positive; lower thresholds add
    # false positives, while higher thresholds miss one fraud.
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.7, 0.5, 0.1])

    threshold = select_threshold_min_fp(y, scores, min_recall=0.8)

    assert threshold == pytest.approx(0.6)
    predictions = scores >= threshold
    assert np.sum((predictions == 1) & (y == 0)) == 1
    assert np.sum((predictions == 1) & (y == 1)) == 2

#%%
def test_threshold_can_trade_recall_for_zero_false_positives():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.7, 0.5, 0.1])

    threshold = select_threshold_min_fp(y, scores, min_recall=0.5)

    assert threshold == pytest.approx(0.9)

#%%
def test_card_precision_top_k_groups_multiple_transactions_from_same_card():
    y = np.array([1, 0, 0, 1, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.9, 0.8])
    customers = np.array(["A", "A", "B", "C", "A", "D", "E"])
    timestamps = np.array(
        [
            "2026-01-01 09:00", "2026-01-01 10:00", "2026-01-01 11:00", "2026-01-01 12:00",
            "2026-01-02 09:00", "2026-01-02 10:00", "2026-01-02 11:00",
        ],
        dtype="datetime64[m]",
    )

    # Day 1: card A (fraud, max score .9) and B (genuine, .7) are top 2 -> .5
    # Day 2: cards D and E are top 2 and both fraudulent -> 1.0
    # Mean CP@2 = .75
    result = card_precision_top_k(
        y, scores, customers, timestamps, top_k=2
    )

    assert result == pytest.approx(0.75)

#%%
def test_decision_inputs_are_validated():
    with pytest.raises(ValueError):
        select_threshold_min_fp([0, 1], [0.1, 0.9], min_recall=0.0)
    with pytest.raises(ValueError):
        card_precision_top_k([0], [0.1], [1], ["2026-01-01"], top_k=0)