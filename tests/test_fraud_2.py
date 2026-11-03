"""Smoke tests for the section 7.2 fraud sweep (experiments/fraud_2), on the small simulated dataset."""

import json

import pytest

from experiments.fraud_2 import run

SMALL = ["--size", "small", "--feature-set", "1"]


@pytest.fixture
def small_data(monkeypatch):
    monkeypatch.setattr(run, "DATA", SMALL)


def test_jobs_file_is_up_to_date(tmp_path):
    run.jobs(tmp_path / "jobs.txt")
    assert (tmp_path / "jobs.txt").read_text() == run.JOBS.read_text()


def test_readout_variants_match_numpy_backend(small_data):
    run.check()


def test_qrc_run_writes_every_readout_variant(small_data, tmp_path):
    (tmp_path / "runs" / "scores").mkdir(parents=True)
    run.run_qrc("tfim", 2, 42, save_scores=True, results=tmp_path)

    records = json.loads((tmp_path / "runs" / "qrc_tfim_L2_s42.json").read_text())
    assert {(r["observables"], r["budget"]) for r in records} == {
        (s, b) for s in run.SETS for b in run.BUDGETS
    }
    assert {r["n_features"] for r in records} == {7, 45}
    assert len(list((tmp_path / "runs" / "scores").glob("*.npz"))) == len(records)


def test_classical_run_goes_through_the_cli(small_data, tmp_path):
    (tmp_path / "runs").mkdir()
    run.run_classical("baseline", 2, 42, results=tmp_path)

    (record,) = json.loads((tmp_path / "runs" / "baseline_L2.json").read_text())
    assert 0.0 <= record["metrics"]["test"]["average_precision"] <= 1.0
    assert (tmp_path / "cli_runs" / "baseline_L2" / "summary.csv").exists()
