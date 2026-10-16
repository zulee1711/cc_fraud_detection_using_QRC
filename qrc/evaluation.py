"""Evaluation broken down by fraud category.

The simulator's ``TX_FRAUD_SCENARIO`` says how a fraud was *generated*, not how hard it
is to *detect*: scenario 3 multiplies amounts by 5, so about half of its frauds end up
above the scenario-1 threshold and are as easy to catch as scenario 1. The categories
here are defined by what the model can observe instead:

    amount    any fraud with ``TX_AMOUNT > AMOUNT_THRESHOLD`` (no genuine transaction
              exceeds it, so the amount alone gives it away)
    terminal  a scenario-2 fraud below the threshold: needs the terminal's history
    customer  a scenario-3 fraud below the threshold: needs the customer's history

The categories are evaluation metadata only. They are derived from the label and must
never be used as a model feature.
"""

import numpy as np
from sklearn.metrics import average_precision_score

# Scenario-1 threshold of the simulator (``qrc.datasets.create_dataset.add_frauds``)
AMOUNT_THRESHOLD = 220
CATEGORIES = ("amount", "terminal", "customer")
LEGIT = "legit"


def categorize(fraud, scenario, amount, threshold=AMOUNT_THRESHOLD):
    """Assign each transaction to ``"legit"`` or one of ``CATEGORIES``.

    Args:
        fraud (array): ``TX_FRAUD``, 0 or 1.
        scenario (array): ``TX_FRAUD_SCENARIO``, 0 to 3.
        amount (array): ``TX_AMOUNT`` as the model sees it (after the scenario-3 ×5).
        threshold (float): the simulator's scenario-1 amount threshold.

    Returns:
        ndarray: one category string per transaction.
    """
    fraud, scenario, amount = (np.asarray(a) for a in (fraud, scenario, amount))
    is_fraud = fraud == 1
    large = amount > threshold

    category = np.full(len(fraud), LEGIT, dtype="U8")
    category[is_fraud & large] = "amount"
    category[is_fraud & ~large & (scenario == 2)] = "terminal"
    category[is_fraud & ~large & (scenario == 3)] = "customer"

    unassigned = is_fraud & (category == LEGIT)
    if unassigned.any():
        raise ValueError(
            f"{unassigned.sum()} frauds fit no category "
            f"(scenarios {sorted(set(scenario[unassigned]))})"
        )
    return category


def category_metrics(labels, scores, category):
    """Average precision of each fraud category against all genuine transactions.

    Frauds of the other categories are left out of the ranking, so each number answers
    "can the model separate this kind of fraud from normal traffic?". When fraud is rare
    (below 1% per category on the shared dataset), a random ranking scores close to 0 in
    every category, so the APs share one 0-to-1 scale.

    Returns:
        dict: ``average_precision_{c}`` for every category; NaN when a category has no
        frauds.
    """
    labels, scores, category = (np.asarray(a) for a in (labels, scores, category))
    legit = category == LEGIT
    metrics = {}
    for name in CATEGORIES:
        positives = category == name
        subset = legit | positives
        metrics[f"average_precision_{name}"] = (
            float(average_precision_score(labels[subset], scores[subset]))
            if positives.any() else float("nan")
        )
    return metrics
