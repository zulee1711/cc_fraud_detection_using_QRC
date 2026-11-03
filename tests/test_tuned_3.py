"""Smoke tests for the equal-budget tuning study (experiments/tuned_3), on the small simulated dataset."""

import json

import numpy as np
import pytest

from experiments.fraud_2 import run as fraud
from experiments.tuned_3 import run
from qrc.hamiltonians import HamiltonianTopology, tfim

pytest.importorskip("optuna")


@pytest.fixture
def small_data(monkeypatch):
    monkeypatch.setattr(fraud, "DATA", ["--size", "small", "--feature-set", "1"])
    monkeypatch.setattr(run, "_DATA", {})


def test_field_one_is_the_cli_tfim():
    H = run.field_tfim(1.0)(4, topology=HamiltonianTopology.LINEAR, rng=42)
    reference = tfim(4, topology=HamiltonianTopology.LINEAR, rng=42)
    assert H.ops.paulis == reference.ops.paulis
    np.testing.assert_allclose(H.ops.coeffs, reference.ops.coeffs)


@pytest.mark.parametrize("model", ["esn", "tfim"])
def test_evaluate_writes_the_best_setting(small_data, tmp_path, model):
    # a one-trial study at the untuned setting, standing in for `tune`
    study = run.study(model, tmp_path)
    study.enqueue_trial(run.REFERENCE[model])
    trial = study.ask()
    run.suggest(trial, model)
    study.tell(trial, 0.5)
    (tmp_path / "runs" / "scores").mkdir(parents=True)

    run.evaluate(model, 42, tmp_path)

    records = json.loads((tmp_path / "runs" / f"{model}_tuned_s42.json").read_text())
    budgets = list(fraud.BUDGETS) if model == "tfim" else [None]
    assert [r["budget"] for r in records] == budgets
    assert all(r["params"] == run.REFERENCE[model] and r["n_features"] == run.N_UNITS for r in records)
    assert len(list((tmp_path / "runs" / "scores").glob("*.npz"))) == len(budgets)
