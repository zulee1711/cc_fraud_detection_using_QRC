"""Small-scale runs of the benchmarks through the full QRC pipeline.

The sizes are cut down from ``experiments.run_benchmarks`` so the whole file
runs in seconds; they are big enough to tell a working reservoir from noise,
not to measure one.
"""

import sys

import numpy as np
import pytest

from experiments.run_benchmarks import (
    build_protocol,
    main,
    run_ipc_benchmark,
    run_memory_benchmark,
    run_narma_benchmark,
    run_parity_benchmark,
)
from qrc.benchmarks import MemoryTask
from qrc.readout import RegressionReadout

SMALL = dict(window_length=4, length=150, n_mem_qubits=2, depth=1)


def _sweep_entries(results):
    return {k: v for k, v in results.items() if k != "total_capacity"}


@pytest.fixture(scope="module")
def memory_results():
    return run_memory_benchmark(**SMALL)


def test_memory_benchmark_runs(memory_results):
    entries = _sweep_entries(memory_results)
    assert sorted(entries) == list(range(SMALL["window_length"]))
    for m in entries.values():
        assert np.isfinite(m["nmse"])
        assert 0.0 <= m["capacity"] <= 1.0
    assert 0.0 <= memory_results["total_capacity"] <= SMALL["window_length"]


def test_parity_benchmark_runs():
    results = run_parity_benchmark(**SMALL)
    entries = _sweep_entries(results)
    assert sorted(entries) == list(range(1, SMALL["window_length"] + 1))
    for m in entries.values():
        assert 0.0 <= m["capacity"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0


def test_narma_benchmark_runs():
    metrics = run_narma_benchmark(window_length=10, length=400, n_mem_qubits=2, depth=1)
    assert np.isfinite(metrics["nmse"])
    assert np.isfinite(metrics["linear_baseline_nmse"])


def test_ipc_benchmark_runs_with_multi_output_readout():
    metrics = run_ipc_benchmark(
        length=120,
        window_length=4,
        max_degree=2,
        max_delay=3,
        washout=10,
        seed=42,
        n_mem_qubits=2,
        depth=1,
    )
    assert set(metrics) == {"degree_1", "degree_2", "total_capacity"}
    assert all(np.isfinite(capacity) and capacity >= 0 for capacity in metrics.values())
    assert metrics["total_capacity"] == pytest.approx(
        metrics["degree_1"] + metrics["degree_2"]
    )


@pytest.mark.parametrize(
    ("hamiltonian", "protocol_argument"),
    [("random", "depth"), ("tfim", "steps")],
)
def test_ipc_cli_maps_reservoir_depth_to_setup(
    monkeypatch, hamiltonian, protocol_argument
):
    calls = []

    def fake_run_ipc_benchmark(**kwargs):
        calls.append(kwargs)
        return {"total_capacity": 0.0}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks",
            "--tasks",
            "ipc",
            "--backend",
            "estimator-exact",
            "--hamiltonian",
            hamiltonian,
            "--ipc-reservoir-depth",
            "4",
            "--observables",
            "XYZ+XYZ-2body",
        ],
    )
    monkeypatch.setattr(
        "experiments.run_benchmarks.run_ipc_benchmark", fake_run_ipc_benchmark
    )

    main()

    assert calls[0][protocol_argument] == 4
    assert calls[0]["observables_spec"] == "XYZ+XYZ-2body"


def test_build_hamiltonian_protocol_uses_requested_trotter_steps():
    protocol = build_protocol(hamiltonian="tfim", steps=4)

    assert protocol.reservoir.steps == 4


@pytest.mark.parametrize("backend", ["gaussian", "shots"])
def test_benchmarks_run_with_finite_precision(backend, memory_results):
    noisy = run_memory_benchmark(backend=backend, precision=0.05, **SMALL)
    assert sorted(_sweep_entries(noisy)) == sorted(_sweep_entries(memory_results))
    assert 0.0 <= noisy["total_capacity"] <= SMALL["window_length"]
    assert noisy["total_capacity"] != memory_results["total_capacity"]  # the noise reaches the features


def test_memory_task_is_solvable_without_a_reservoir():
    """Control for the test below: the readout on the raw windows solves every
    delay, so a reservoir that fails it is losing the inputs."""
    L = SMALL["window_length"]
    for delay in range(L):
        (X_train, y_train), (X_test, y_test) = MemoryTask(delay=delay, length=150, window_length=L, seed=42).generate()
        readout = RegressionReadout()
        readout.fit(X_train[:, :, 0], y_train)
        metrics, _ = readout.evaluate(X_test[:, :, 0], y_test)
        assert metrics["capacity"] == pytest.approx(1.0)


def test_reservoir_recalls_the_current_input(memory_results):
    assert memory_results[0]["capacity"] > 0.5
