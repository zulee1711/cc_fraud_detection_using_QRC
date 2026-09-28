"""Small-scale runs of the benchmarks through the full QRC pipeline.

The sizes are cut down from ``experiments.run_benchmarks`` so the whole file
runs in seconds; they are big enough to tell a working reservoir from noise,
not to measure one.
"""

import numpy as np
import pytest

from experiments.run_benchmarks import run_memory_benchmark, run_narma_benchmark, run_parity_benchmark
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


def test_memory_task_is_solvable_without_a_reservoir():
    """Control for the test below: the readout on the raw windows solves every
    delay, so a reservoir that fails it is losing the inputs."""
    L = SMALL["window_length"]
    for delay in range(L):
        (X_train, y_train), (X_test, y_test) = MemoryTask(delay=delay, length=150, window_length=L, seed=42).generate()
        metrics, _ = RegressionReadout().fit_evaluate(X_train[:, :, 0], y_train, X_test[:, :, 0], y_test)
        assert metrics["capacity"] == pytest.approx(1.0)


@pytest.mark.xfail(
    strict=True,
    reason="StatevectorBackend never resets the input qubits between steps (TODO in run_window), "
           "so the reservoir loses even the current input. Remove this mark once that is fixed.",
)
def test_reservoir_recalls_the_current_input(memory_results):
    assert memory_results[0]["capacity"] > 0.5
