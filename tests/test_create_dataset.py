"""The fraud simulator."""

import pytest

from qrc.datasets.create_dataset import add_frauds, generate_dataset


@pytest.fixture(scope="module")
def simulation():
    # enough customers that a few compromised per day stay a small share
    return generate_dataset(n_customers=300, n_terminals=600, nb_days=60, default_random_state=0)


def scenario_3_frauds(simulation, **kwargs):
    customers, terminals, transactions = simulation
    transactions = add_frauds(customers, terminals, transactions.copy(), **kwargs)
    return (transactions.TX_FRAUD_SCENARIO == 3).sum()


def test_default_compromises_three_customers_per_day(simulation):
    assert scenario_3_frauds(simulation) == scenario_3_frauds(simulation, n_compromised_customers=3)


def test_more_compromised_customers_give_more_scenario_3_frauds(simulation):
    assert scenario_3_frauds(simulation, n_compromised_customers=9) > 2 * scenario_3_frauds(simulation)
