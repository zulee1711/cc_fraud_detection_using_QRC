"""Smoke tests for the experiment runner, on the smallest simulation preset."""

import csv
import json

import numpy as np
import pytest

import experiments.cli as cli
from experiments.cli import main
from qrc.encodings import DenseAngleEncoding


def _metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text())


@pytest.mark.parametrize("command", [["baseline"], ["qrc", "--backend", "estimator-exact"]])
def test_run_saves_results(tmp_path, command):
    run_dir = main([*command, "--size", "small", "--output-dir", str(tmp_path)])

    assert {p.name for p in run_dir.iterdir()} == {"config.json", "metrics.json", "scores.npz"}
    for split in ("validation", "test"):
        assert 0.0 <= _metrics(run_dir)[split]["pr_auc"] <= 1.0

    with open(tmp_path / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["run"] for row in rows] == [run_dir.name]


def test_saved_simulation_reloads_identically(tmp_path):
    """A simulated dataset written with --save-data gives the same run when loaded back."""
    data_dir, output_dir = tmp_path / "data", tmp_path / "runs"
    simulated = main(["baseline", "--size", "small", "--save-data", str(data_dir), "--output-dir", str(output_dir)])
    loaded = main(["baseline", "--data", "load", "--data-dir", str(data_dir), "--output-dir", str(output_dir)])

    def splits(run_dir):
        return json.loads((run_dir / "config.json").read_text())["splits"]

    assert splits(loaded) == splits(simulated)
    for split in ("validation", "test"):
        assert _metrics(loaded)[split] == pytest.approx(_metrics(simulated)[split])


def test_summary_records_reservoir_and_metrics(tmp_path):
    """A qrc row in summary.csv carries the resolved reservoir settings and every metric."""
    run_dir = main(["qrc", "--size", "small", "--n-mem-qubits", "1", "--observables", "XYZ",
                    "--output-dir", str(tmp_path)])

    with open(tmp_path / "summary.csv", newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["run"] == run_dir.name
    assert (row["n_input_qubits"], row["n_mem_qubits"], row["n_qubits"]) == ("7", "1", "8")
    assert (row["observables"], row["n_observables"], row["seed"]) == ("XYZ", "24", "42")
    assert row["backend_precision"] == ""  # unused by the exact backend
    for split in ("validation", "test"):
        for metric in ("pr_auc", "roc_auc", "f1_score", "precision", "recall"):
            assert row[f"{split}_{metric}"] != ""


def test_qrc_cli_selects_dense_encoding_when_qubits_are_fewer_than_features(monkeypatch):
    args = cli.build_parser().parse_args(
        ["qrc", "--n-input-qubits", "4", "--n-mem-qubits", "1"]
    )
    protocol_args = []

    class RecordingProtocol:
        def __init__(self, encoder, reservoir, observables, backend):
            protocol_args.append((encoder, reservoir, observables, backend))

        def run(self, windows):
            return np.zeros((len(windows), 1))

    monkeypatch.setattr(cli, "QRCProtocol", RecordingProtocol)
    monkeypatch.setattr(cli, "make_backend", lambda *args: args)
    data = {f"X_{split}": np.zeros((1, 1, 7)) for split in ("train", "validation", "test")}

    cli.qrc_features(data, args)

    assert len(protocol_args) == 1
    encoder = protocol_args[0][0]
    assert isinstance(encoder, DenseAngleEncoding)
    assert encoder.num_qubits == 4
