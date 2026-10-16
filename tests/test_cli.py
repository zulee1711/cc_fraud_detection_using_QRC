"""Smoke tests for the experiment runner, on the smallest simulation preset."""

import csv
import json

import numpy as np
import pytest

import experiments.cli as cli
from experiments.cli import main
from qrc.encodings import DenseAngleEncoding
from qrc.reservoirs import HamiltonianReservoir, RandomCircuitReservoir


def _metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text())


@pytest.mark.parametrize(
    "command",
    [
        ["baseline"],
        ["lstm", "--epochs", "2", "--hidden-size", "4", "--batch-size", "1024"],
        ["qrc", "--backend", "estimator-exact"],
    ],
)
def test_run_saves_results(tmp_path, command):
    if command[0] == "lstm":
        pytest.importorskip("torch")
    run_dir = main([*command, "--size", "small", "--output-dir", str(tmp_path)])

    assert {p.name for p in run_dir.iterdir()} == {
        "config.json",
        "metrics.json",
        "scores.npz",
    }
    for split in ("validation", "test"):
        assert 0.0 <= _metrics(run_dir)[split]["average_precision"] <= 1.0
        assert "average_precision_customer" in _metrics(run_dir)[split]

    saved = np.load(run_dir / "scores.npz")
    assert np.array_equal(saved["category_test"] != "legit", saved["y_test"] == 1)
    assert len(saved["ids_test"]) == len(saved["scores_test"])

    with open(tmp_path / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["run"] for row in rows] == [run_dir.name]
    assert sum(int(rows[0][f"test_frauds_{c}"]) for c in ("amount", "terminal", "customer")) == int(rows[0]["test_frauds"])


def test_saved_simulation_reloads_identically(tmp_path):
    """A simulated dataset written with --save-data gives the same run when loaded back."""
    data_dir, output_dir = tmp_path / "data", tmp_path / "runs"
    simulated = main(
        [
            "baseline",
            "--size",
            "small",
            "--save-data",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    loaded = main(
        [
            "baseline",
            "--data",
            "load",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    def splits(run_dir):
        return json.loads((run_dir / "config.json").read_text())["splits"]

    assert splits(loaded) == splits(simulated)
    for split in ("validation", "test"):
        assert _metrics(loaded)[split] == pytest.approx(_metrics(simulated)[split])


def test_summary_records_reservoir_and_metrics(tmp_path):
    """A qrc row in summary.csv carries the resolved reservoir settings and every metric."""
    run_dir = main(
        [
            "qrc",
            "--size",
            "small",
            "--n-mem-qubits",
            "1",
            "--observables",
            "XYZ",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with open(tmp_path / "summary.csv", newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["run"] == run_dir.name
    assert (row["n_input_qubits"], row["n_mem_qubits"], row["n_qubits"]) == (
        "7",
        "1",
        "8",
    )
    assert (row["observables"], row["n_observables"], row["seed"]) == (
        "XYZ",
        "24",
        "42",
    )
    assert row["backend_precision"] == ""  # unused by the exact backend
    assert row["hamiltonian"] == "random"
    assert row["topology"] == ""
    for split in ("validation", "test"):
        for metric in cli.METRIC_NAMES:
            assert row[f"{split}_{metric}"] != ""


def test_qrc_cli_selects_dense_encoding_when_qubits_are_fewer_than_features(
    monkeypatch,
):
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
    data = {
        f"X_{split}": np.zeros((1, 1, 7)) for split in ("train", "validation", "test")
    }

    cli.qrc_features(data, args)

    assert len(protocol_args) == 1
    encoder = protocol_args[0][0]
    assert isinstance(encoder, DenseAngleEncoding)
    assert encoder.num_qubits == 4


def test_qrc_cli_runs_with_dense_encoding(tmp_path):
    run_dir = main(
        [
            "qrc",
            "--size",
            "small",
            "--n-input-qubits",
            "6",
            "--n-mem-qubits",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    config = json.loads((run_dir / "config.json").read_text())
    assert config["reservoir"]["n_input_qubits"] == 6
    assert 0.0 <= _metrics(run_dir)["test"]["average_precision"] <= 1.0


def test_qrc_cli_runs_with_fewer_input_qubits(tmp_path):
    """QRC should run when dense encoding is required."""
    run_dir = main(
        [
            "qrc",
            "--size",
            "small",
            "--n-input-qubits",
            "6",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert run_dir.exists()
    assert {p.name for p in run_dir.iterdir()} == {
        "config.json",
        "metrics.json",
        "scores.npz",
    }

    metrics = _metrics(run_dir)

    for split in ("validation", "test"):
        assert 0.0 <= metrics[split]["average_precision"] <= 1.0



def test_qrc_cli_selects_hamiltonian_and_topology(monkeypatch):
    args = cli.build_parser().parse_args(
        [
            "qrc",
            "--n-input-qubits",
            "7",
            "--n-mem-qubits",
            "1",
            "--hamiltonian",
            "tfim",
            "--topology",
            "ring",
        ]
    )
    captured = []

    class RecordingProtocol:
        def __init__(self, *protocol_args):
            captured.append(protocol_args[1])

        def run(self, windows):
            return np.zeros((len(windows), 1))

    monkeypatch.setattr(cli, "QRCProtocol", RecordingProtocol)
    monkeypatch.setattr(cli, "make_backend", lambda *args: args)
    data = {
        f"X_{split}": np.zeros((1, 1, 7)) for split in ("train", "validation", "test")
    }

    _, reservoir_info = cli.qrc_features(data, args)

    assert len(captured) == 1
    assert isinstance(captured[0], HamiltonianReservoir)
    assert captured[0].hamiltonian.num_qubits == 8
    assert len(captured[0].hamiltonian.params) == 8
    assert reservoir_info["hamiltonian"] == "tfim"
    assert reservoir_info["topology"] == "ring"


def test_qrc_cli_keeps_random_reservoir_as_default(monkeypatch):
    args = cli.build_parser().parse_args(["qrc", "--n-input-qubits", "7"])
    captured = []

    class RecordingProtocol:
        def __init__(self, *protocol_args):
            captured.append(protocol_args[1])

        def run(self, windows):
            return np.zeros((len(windows), 1))

    monkeypatch.setattr(cli, "QRCProtocol", RecordingProtocol)
    monkeypatch.setattr(cli, "make_backend", lambda *args: args)
    data = {
        f"X_{split}": np.zeros((1, 1, 7)) for split in ("train", "validation", "test")
    }

    _, reservoir_info = cli.qrc_features(data, args)

    assert isinstance(captured[0], RandomCircuitReservoir)
    assert reservoir_info["hamiltonian"] == "random"
    assert reservoir_info["topology"] == ""


def test_cli_runs_on_raw_amount_windows(tmp_path):
    """Arm B's input: the raw log amount only, no engineered history."""
    run_dir = main(["baseline", "--size", "small", "--feature-set", "1", "--window-length", "4",
                    "--output-dir", str(tmp_path)])

    with open(tmp_path / "summary.csv", newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["run"] == run_dir.name
    assert (row["feature_set"], row["n_features"]) == ("1", "1")


def test_evaluation_cli_arguments_are_exposed():
    args = cli.build_parser().parse_args(
        ["baseline", "--min-recall", "0.9", "--top-k", "25"]
    )

    assert args.min_recall == pytest.approx(0.9)
    assert args.top_k == 25


def test_fit_and_evaluate_selects_threshold_on_validation(monkeypatch):
    calls = []

    class RecordingReadout:
        def __init__(self, alpha):
            calls.append(("init", alpha))
            self.decision_threshold = 0.73

        def fit(self, X, y):
            calls.append(("fit", X.copy(), y.copy()))

        def select_decision_threshold(self, X, y, *, min_recall):
            calls.append(("select", X.copy(), y.copy(), min_recall))
            return self.decision_threshold

        def evaluate(self, X, y):
            calls.append(("evaluate", X.copy(), y.copy()))
            scores = np.full(len(y), self.decision_threshold)
            metrics = {
                "threshold_used": self.decision_threshold,
                "average_precision": 0.5,
                "roc_auc": 0.5,
                "f1_score": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "false_positive_rate": 0.5,
                "matthews_corr_coef": 0.0,
                "balanced_accuracy": 0.5,
            }
            return metrics, y, np.ones(len(y), dtype=int), scores

    monkeypatch.setattr(cli, "ClassicalReadout", RecordingReadout)

    features = {
        "train": np.array([[1.0], [2.0]]),
        "validation": np.array([[3.0], [4.0]]),
        "test": np.array([[5.0], [6.0]]),
    }
    data = {
        "y_train": np.array([0, 1]),
        "y_validation": np.array([1, 0]),
        "y_test": np.array([0, 1]),
    }

    metrics, _ = cli.fit_and_evaluate(
        features,
        data,
        alpha=2.0,
        min_recall=0.9,
    )

    assert calls[0] == ("init", 2.0)
    assert calls[1][0] == "fit"
    np.testing.assert_array_equal(calls[1][1], features["train"])
    np.testing.assert_array_equal(calls[1][2], data["y_train"])

    assert calls[2][0] == "select"
    np.testing.assert_array_equal(calls[2][1], features["validation"])
    np.testing.assert_array_equal(calls[2][2], data["y_validation"])
    assert calls[2][3] == pytest.approx(0.9)

    assert [call[0] for call in calls[3:]] == ["evaluate", "evaluate"]
    assert metrics["validation"]["threshold_used"] == pytest.approx(0.73)
    assert metrics["test"]["threshold_used"] == pytest.approx(0.73)


def test_add_operational_metrics_uses_customer_and_timestamp_metadata(monkeypatch):
    calls = []

    def fake_card_precision(y, scores, customers, timestamps, *, top_k):
        calls.append((y.copy(), scores.copy(), customers.copy(), timestamps.copy(), top_k))
        return 0.4

    monkeypatch.setattr(cli, "card_precision_top_k", fake_card_precision)

    metrics = {"validation": {}, "test": {}}
    scores = {
        "validation": np.array([0.2, 0.8]),
        "test": np.array([0.3, 0.9]),
    }
    data = {
        "y_validation": np.array([0, 1]),
        "y_test": np.array([0, 1]),
        "groups_validation": np.array([10, 11]),
        "groups_test": np.array([20, 21]),
        "timestamps_validation": np.array(
            ["2026-01-01", "2026-01-01"], dtype="datetime64[D]"
        ),
        "timestamps_test": np.array(
            ["2026-01-02", "2026-01-02"], dtype="datetime64[D]"
        ),
    }

    cli.add_operational_metrics(metrics, scores, data, top_k=25)

    assert len(calls) == 2
    assert all(call[-1] == 25 for call in calls)
    assert metrics["validation"]["card_precision_top_k"] == pytest.approx(0.4)
    assert metrics["test"]["card_precision_top_k"] == pytest.approx(0.4)
