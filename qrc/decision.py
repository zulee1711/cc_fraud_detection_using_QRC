#%%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

#%%
def select_threshold_min_fp(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_recall: float,
):
    """
    Select the threshold that minimizes false positives while ensuring a minimum recall.
    """
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if not len(y_true):
        raise ValueError("cannot select a threshold from an empty validation set")
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1]")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if np.sum(y_true == 1) == 0:
        raise ValueError("validation set must contain at least one fraud")

    # prediction changes only when threshold crosses an observed score, candidates are the unique scores
    candidates = np.unique(scores)
    feasible = []

    for threshold in candidates:
        y_pred = scores >= threshold
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall < min_recall:
            continue

        false_positives = int(np.sum((y_pred == 1) & (y_true == 0)))
        feasible.append((false_positives, -recall, -float(threshold), float(threshold)))

    if not feasible:
        raise ValueError(
            f"no threshold satisfies the minimum recall requirement ({min_recall})"
        )

    return min(feasible)[-1]

#%%
def card_precision_top_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    customer_ids: np.ndarray,
    timestamps: np.ndarray,
    *,
    top_k: int,
):
    """
    Return mean daily Card Precision@K
    """
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    customer_ids = np.asarray(customer_ids).reshape(-1)
    timestamps = np.asarray(timestamps).reshape(-1)

    lengths = {len(y_true), len(scores), len(customer_ids), len(timestamps)}
    if len(lengths) != 1:
        raise ValueError("labels, scores, customer_ids and timestamps must have the same length")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not len(y_true):
        return float("nan")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")

    frame = pd.DataFrame(
        {
            "label": y_true.astype(int),
            "score": scores,
            "customer_id": customer_ids,
            "timestamp": pd.to_datetime(timestamps),
        }
    )
    frame["day"] = frame["timestamp"].dt.normalize()

    per_card = (
        frame.groupby(["day", "customer_id"], sort=True, as_index=False)
        .agg(label=("label", "max"), score=("score", "max"))
    )

    daily_precisions = []
    for _, day_cards in per_card.groupby("day", sort=True):
        alerts = day_cards.nlargest(top_k, "score")
        if len(alerts):
            daily_precisions.append(float(alerts["label"].mean()))

    return float(np.mean(daily_precisions)) if daily_precisions else float("nan")