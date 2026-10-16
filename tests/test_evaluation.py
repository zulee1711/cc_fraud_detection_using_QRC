import math

import numpy as np
import pytest

from qrc.evaluation import CATEGORIES, categorize, category_metrics


def test_categorize_by_observable_amount():
    fraud = [0, 1, 1, 1, 1, 1]
    scenario = [0, 1, 2, 2, 3, 3]
    amount = [500.0, 300.0, 50.0, 400.0, 60.0, 1000.0]

    assert categorize(fraud, scenario, amount).tolist() == [
        "legit", "amount", "terminal", "amount", "customer", "amount",
    ]


def test_categorize_rejects_unexplained_frauds():
    with pytest.raises(ValueError, match="fit no category"):
        categorize([1], [1], [100.0])


def test_perfect_scores_give_unit_ap_in_every_category():
    category = np.array(["legit", "legit", "amount", "terminal", "customer", "legit"])
    labels = (category != "legit").astype(int)
    scores = labels.astype(float)

    metrics = category_metrics(labels, scores, category)

    for name in CATEGORIES:
        assert metrics[f"average_precision_{name}"] == 1.0


def test_other_categories_do_not_enter_the_ranking():
    # the terminal fraud outranks the customer fraud, but is not part of the customer subset
    category = np.array(["terminal", "customer", "legit", "legit"])
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])

    assert category_metrics(labels, scores, category)["average_precision_customer"] == 1.0


def test_missing_category_is_nan():
    category = np.array(["legit", "amount"])
    metrics = category_metrics([0, 1], [0.1, 0.9], category)

    assert math.isnan(metrics["average_precision_customer"])
    assert math.isnan(metrics["average_precision_terminal"])
