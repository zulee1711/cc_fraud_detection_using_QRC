#%%
from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd

#%%
def _validate_binary_inputs(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    allow_empty: bool = False,
):
    """
    Validate labels and model scores
    """
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if len(y_true) != len(scores):
        raise ValueError(
            "y_true and scores must have the same length"
        )

    if len(y_true) == 0:
        if allow_empty:
            return (
                y_true.astype(np.int8, copy=False),
                scores,
            )

        raise ValueError(
            "inputs must not be empty"
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "scores must be finite"
        )

    if not np.isin(y_true, (0, 1)).all():
        raise ValueError(
            "y_true must contain only binary labels 0 and 1"
        )

    return (
        y_true.astype(np.int8, copy=False),
        scores,
    )

#%%
def _threshold_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
):
    """
        Build the validation threshold curve once.

        Thresholds are returned in descending order.
        For each threshold, ``false_positives`` and ``recall`` correspond to predictions defined as::
            scores >= threshold

        Equal scores are treated as one threshold candidate because all
        observations with the same score must enter the positive class together.
    """
    y_true, scores = _validate_binary_inputs(y_true, scores)

    n_positive = int(y_true.sum())

    if n_positive == 0:
        raise ValueError(
            "validation set must contain at least one fraud"
        )

    # Sort observations from highest predicted fraud score to lowest.
    order = np.argsort(
        -scores,
        kind="stable",
    )

    sorted_scores = scores[order]
    sorted_y = y_true[order]

    # Running confusion-matrix counts as the threshold decreases.
    true_positives = np.cumsum(
        sorted_y,
        dtype=np.int64,
    )

    false_positives = np.cumsum(
        1 - sorted_y,
        dtype=np.int64,
    )

    # Predictions use scores >= threshold
    distinct_end = np.r_[
        sorted_scores[:-1] != sorted_scores[1:],
        True,
    ]

    thresholds = sorted_scores[distinct_end]
    true_positives = true_positives[distinct_end]
    false_positives = false_positives[distinct_end]

    recall = true_positives / n_positive

    return (thresholds, false_positives, recall)

#%%
def _validate_min_recalls(
    min_recalls: Iterable[float],
):
    """
    Validate and normalize minimum-recall constraints.
    """
    min_recalls = np.asarray(
        list(min_recalls),
        dtype=float,
    ).reshape(-1)

    if len(min_recalls) == 0:
        raise ValueError(
            "min_recalls must contain at least one value"
        )

    if not np.all(np.isfinite(min_recalls)):
        raise ValueError(
            "min_recalls must contain only finite values"
        )

    if np.any(
        (min_recalls <= 0.0) | (min_recalls > 1.0)
    ):
        raise ValueError(
            "every min_recall must be in (0, 1]"
        )

    # Sorting and removing duplicates avoids unnecessary repeated work.
    return np.unique(min_recalls)

#%%
def select_thresholds_min_fp(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_recalls: Iterable[float],
):
    """
    Select decision thresholds for multiple minimum-recall constraints.

    The threshold curve is calculated only once.

    For every minimum-recall constraint, the selected threshold:

    1. satisfies ``recall >= min_recall``
    2. minimizes false positives
    3. if false positives tie, maximizes recall
    4. if still tied, prefers the higher threshold
    """
    min_recalls = _validate_min_recalls(min_recalls)

    (thresholds, false_positives, recall) = _threshold_curve(y_true, scores)

    # Recall is non-decreasing as the threshold decreases
    first_feasible = np.searchsorted(
        recall,
        min_recalls,
        side="left",
    )

    if np.any(first_feasible >= len(thresholds)):
        invalid = min_recalls[
            first_feasible >= len(thresholds)
            ]

        raise ValueError(
            "no threshold satisfies one or more "
            f"minimum recall requirements: {invalid.tolist()}"
        )

    # Because FP is non-decreasing as threshold decreases,
    # 1st feasible threshold has min FP
    minimum_fp = false_positives[first_feasible]

    best_indices = (
            np.searchsorted(
                false_positives,
                minimum_fp,
                side="right",
            )
            - 1
    )

    selected_thresholds = thresholds[best_indices]

    return {
        float(min_recall): float(threshold)
        for min_recall, threshold in zip(
            min_recalls,
            selected_thresholds,
            strict=True,
        )
    }

#%%
def select_threshold_min_fp(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_recall: float,
):
    """
    Select one validation decision threshold.
    """
    result = select_thresholds_min_fp(
        y_true,
        scores,
        min_recalls=[min_recall],
    )

    return result[float(min_recall)]


#%%
def select_thresholds_max_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    max_fprs: Iterable[float],
):
    """
    Select decision thresholds for multiple false-positive-rate budgets.

    The counterpart of ``select_thresholds_min_fp`` with the constraint on the
    other error: for every budget, the selected threshold

    1. satisfies ``false_positive_rate <= max_fpr``
    2. maximizes recall
    3. if recall ties, minimizes false positives (the higher threshold)

    Predictions are ``scores >= threshold``, as in ``select_thresholds_min_fp``.
    A budget too small for any threshold gives ``inf`` (nothing is flagged).
    """
    max_fprs = _validate_min_recalls(max_fprs)  # same checks: finite, in (0, 1]

    (thresholds, false_positives, recall) = _threshold_curve(y_true, scores)

    n_negative = int((np.asarray(y_true).reshape(-1) == 0).sum())

    # FP is non-decreasing as the threshold decreases: the last feasible
    # threshold has the highest recall
    last_feasible = (
            np.searchsorted(
                false_positives,
                max_fprs * n_negative,
                side="right",
            )
            - 1
    )

    # Recall is non-decreasing too: the first threshold reaching that recall
    # has the fewest false positives
    best_indices = np.searchsorted(
        recall,
        recall[np.maximum(last_feasible, 0)],
        side="left",
    )

    return {
        float(max_fpr): (float(thresholds[best]) if last >= 0 else float("inf"))
        for max_fpr, last, best in zip(
            max_fprs,
            last_feasible,
            best_indices,
            strict=True,
        )
    }

#%%
def select_threshold_max_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    max_fpr: float,
):
    """
    Select one validation decision threshold under a false-positive-rate budget.
    """
    result = select_thresholds_max_recall(
        y_true,
        scores,
        max_fprs=[max_fpr],
    )

    return result[float(max_fpr)]


#%%
def _validate_card_inputs(
    y_true: np.ndarray,
    scores: np.ndarray,
    customer_ids: np.ndarray,
    timestamps: np.ndarray,
):
    """
    Validate inputs required for Card Precision@K
    """
    y_true, scores = _validate_binary_inputs(y_true, scores, allow_empty=True,)

    customer_ids = np.asarray(customer_ids).reshape(-1)

    timestamps = np.asarray(timestamps).reshape(-1)

    n = len(y_true)

    if (
        len(customer_ids) != n
        or len(timestamps) != n
    ):
        raise ValueError(
            "labels, scores, customer_ids and timestamps must have the same length"
        )

    if pd.isna(customer_ids).any():
        raise ValueError(
            "customer_ids must not contain missing values"
        )

    days = pd.to_datetime(
        timestamps,
        errors="raise",
    ).normalize()

    if days.isna().any():
        raise ValueError(
            "timestamps must not contain missing values"
        )

    return (
        y_true,
        scores,
        customer_ids,
        days,
    )

#%%
def _daily_card_ranking(
    y_true: np.ndarray,
    scores: np.ndarray,
    customer_ids: np.ndarray,
    timestamps: np.ndarray,
):
    """
    Build the daily customer/card return ranking.

    Transactions are collapsed to one observation per ``(day, customer)``:

    - card label = maximum fraud label for that day;
    - card score = maximum fraud score for that day.

    Cards are then ranked by descending fraud score within each day.
    """
    (y_true, scores, customer_ids, days) = _validate_card_inputs(
        y_true,
        scores,
        customer_ids,
        timestamps,
    )

    if len(y_true) == 0:
        empty = np.array([], dtype=np.int64)

        return (empty, empty, empty, empty)

    day_codes, _ = pd.factorize(days, sort=True)

    customer_codes, _ = pd.factorize(customer_ids, sort=True)

    # Collapse transactions -> one row per (day, customer)
    pair_order = np.lexsort(
        (
            customer_codes,
            day_codes,
        )
    )

    ordered_days = day_codes[pair_order]

    ordered_customers = customer_codes[pair_order]

    ordered_labels = y_true[pair_order]

    ordered_scores = scores[pair_order]

    # Locate the first row of every new (day, customer) pair.
    new_pair = np.r_[
        True,
        (
                ordered_days[1:]
                != ordered_days[:-1]
        )
        |
        (
                ordered_customers[1:]
                != ordered_customers[:-1]
        ),
    ]

    group_starts = np.flatnonzero(new_pair)

    # Maximum label and score for each customer/day.
    card_labels = np.maximum.reduceat(
        ordered_labels,
        group_starts,
    )

    card_scores = np.maximum.reduceat(
        ordered_scores,
        group_starts,
    )

    card_days = ordered_days[group_starts]

    card_customers = ordered_customers[group_starts]

    # Rank cards inside each day
    ranking_order = np.lexsort(
        (
            card_customers,
            -card_scores,
            card_days,
        )
    )

    ranked_days = card_days[
        ranking_order
    ]

    ranked_labels = card_labels[
        ranking_order
    ].astype(
        np.int64,
        copy=False,
    )

    ranked_customers = card_customers[
        ranking_order
    ]

    # Find daily ranking boundaries.
    day_start_mask = np.r_[
        True,
        ranked_days[1:]
        != ranked_days[:-1],
    ]

    day_starts = np.flatnonzero(day_start_mask)

    day_ends = np.r_[
        day_starts[1:],
        len(ranked_days),
    ]

    day_sizes = (day_ends - day_starts)

    return (
        day_starts,
        day_sizes,
        ranked_labels,
        ranked_customers,
    )

#%%
def _validate_top_ks(
    top_ks: Iterable[int],
):
    """
        Validate and normalize Card Precision@K capacities.
        """
    raw_top_ks = list(top_ks)

    if len(raw_top_ks) == 0:
        raise ValueError(
            "top_ks must contain at least one value"
        )

    validated: list[int] = []

    for value in raw_top_ks:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "every top_k must be an integer >= 1"
            )

        integer_value = int(value)

        if integer_value != value:
            raise ValueError(
                "every top_k must be an integer >= 1"
            )

        if integer_value < 1:
            raise ValueError(
                "every top_k must be >= 1"
            )

        validated.append(integer_value)

    return np.unique(
        np.asarray(
            validated,
            dtype=np.int64,
        )
    )

#%%
def card_precision_top_ks(
    y_true: np.ndarray,
    scores: np.ndarray,
    customer_ids: np.ndarray,
    timestamps: np.ndarray,
    *,
    top_ks: Iterable[int],
    remove_detected: bool = True,
):
    """
    Return mean daily Card Precision@K, as defined in the Fraud Detection Handbook
    (Le Borgne et al., "Precision top-k metrics")

    Transactions are first collapsed to one observation per
    ``(day, customer)``:

    - card label = maximum fraud label that day
    - card score = maximum fraud score that day

    Cards are ranked by descending score within each day, and the top K are
    reviewed. Daily precision is:
        fraudulent cards among the top K // K

    K is the daily review capacity, so the denominator stays K on days with
    fewer than K cards.

    With ``remove_detected`` (the handbook default), a fraudulent card found in
    the top K is blocked: it is dropped from every later day, so the same
    compromised card cannot be counted again. Removal depends on K, so each K
    is evaluated separately.

    The final Card Precision@K is the mean of the daily precision values.
    """
    top_ks = _validate_top_ks(top_ks)

    (day_starts, day_sizes, ranked_labels, ranked_customers) = _daily_card_ranking(
        y_true,
        scores,
        customer_ids,
        timestamps,
    )

    if len(day_starts) == 0:
        return {
            int(top_k): float("nan")
            for top_k in top_ks
        }

    n_customers = int(ranked_customers.max()) + 1
    results = {}

    for top_k in top_ks:
        detected = np.zeros(n_customers, dtype=bool)
        daily_precision = np.empty(len(day_starts))

        for day, (start, size) in enumerate(zip(day_starts, day_sizes)):
            customers = ranked_customers[start:start + size]
            labels = ranked_labels[start:start + size]

            remaining = ~detected[customers]
            reviewed = customers[remaining][:top_k]
            reviewed_labels = labels[remaining][:top_k]

            daily_precision[day] = reviewed_labels.sum() / top_k

            if remove_detected:
                detected[reviewed[reviewed_labels == 1]] = True

        results[int(top_k)] = float(daily_precision.mean())

    return results

#%%
def card_precision_top_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    customer_ids: np.ndarray,
    timestamps: np.ndarray,
    *,
    top_k: int,
    remove_detected: bool = True,
):
    """
    Return mean daily Card Precision@K (see `card_precision_top_ks`)
    """
    result = card_precision_top_ks(
        y_true,
        scores,
        customer_ids,
        timestamps,
        top_ks=[top_k],
        remove_detected=remove_detected,
    )

    return result[int(top_k)]