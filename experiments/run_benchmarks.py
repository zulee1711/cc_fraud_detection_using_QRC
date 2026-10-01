"""Run the benchmark tasks through the QRC pipeline.

    python -m experiments.run_benchmarks

The memory and parity benchmarks sweep one task parameter (the delay, the
order). The reservoir is run once per sweep: with a fixed seed every task in the
sweep sees the same windows, only the target changes, so each one just fits a
new readout. NARMA10 is a single task.
"""

import numpy as np

from qrc.backends.statevector import StatevectorBackend
from qrc.benchmarks import MemoryTask, NARMA10Task, ParityTask
from qrc.encodings import AngleEncoding
from qrc.observables import build_observables
from qrc.protocol import QRCProtocol
from qrc.readout import RegressionReadout
from qrc.reservoirs import RandomCircuitReservoir


def build_protocol(n_input_qubits=1, n_mem_qubits=3, depth=2, observables_spec="Z+ZZ", seed=42):
    """The encoder → reservoir → observables stack shared by every benchmark."""
    encoder = AngleEncoding(num_qubits=n_input_qubits)
    reservoir = RandomCircuitReservoir(num_input_qubits=n_input_qubits, num_mem_qubits=n_mem_qubits, depth=depth, rng=seed)
    observables = build_observables(observables_spec, reservoir.num_qubits)
    return QRCProtocol(encoder, reservoir, observables, StatevectorBackend())


def _score_sweep(protocol, tasks, alpha):
    """Run the reservoir on the (shared) windows once, then fit one readout per task.

    Returns:
        list of ``(metrics, y_test, y_pred)``, one per task, in order.
    """
    (X_train, _), (X_test, _) = tasks[0].generate()
    train_outputs = protocol.run(X_train)
    test_outputs = protocol.run(X_test)

    scores = []
    for task in tasks:
        (_, y_train), (_, y_test) = task.generate()
        readout = RegressionReadout(alpha=alpha)
        readout.fit(train_outputs, y_train)
        metrics, y_pred = readout.evaluate(test_outputs, y_test)
        scores.append((metrics, y_test, y_pred))
    return scores


def run_memory_benchmark(window_length=10, length=1000, seed=42, alpha=1e-6, **protocol_kwargs):
    """Score the reservoir on every delay the window allows.

    Returns:
        dict mapping each delay to its readout metrics, plus ``total_capacity``,
        the sum of the per-delay capacities (at most ``window_length``).
    """
    protocol = build_protocol(seed=seed, **protocol_kwargs)
    tasks = [MemoryTask(delay=d, length=length, window_length=window_length, seed=seed)
             for d in range(window_length)]

    results = {task.delay: metrics for task, (metrics, _, _) in zip(tasks, _score_sweep(protocol, tasks, alpha))}
    results["total_capacity"] = sum(results[d]["capacity"] for d in range(window_length))
    return results


def run_parity_benchmark(window_length=10, length=1000, seed=42, alpha=1e-6, **protocol_kwargs):
    """Score the reservoir on every parity order the window allows.

    The readout is the same ridge regression as for the memory task, fitted on
    the 0/1 target; its output is thresholded at 0.5 to get a class, so
    ``accuracy`` is 0.5 at chance level.

    Returns:
        dict mapping each order to its readout metrics (``nmse``, ``capacity``,
        ``accuracy``), plus ``total_capacity``.
    """
    protocol = build_protocol(seed=seed, **protocol_kwargs)
    tasks = [ParityTask(order=n, length=length, window_length=window_length, seed=seed)
             for n in range(1, window_length + 1)]

    results = {}
    for task, (metrics, y_test, y_pred) in zip(tasks, _score_sweep(protocol, tasks, alpha)):
        metrics["accuracy"] = float(np.mean((y_pred >= 0.5) == y_test))
        results[task.order] = metrics
    results["total_capacity"] = sum(results[n]["capacity"] for n in range(1, window_length + 1))
    return results


def run_narma_benchmark(window_length=10, length=2000, seed=42, alpha=1e-6, **protocol_kwargs):
    """Score the reservoir on NARMA10 one-step prediction.

    Returns:
        dict with the readout metrics (``nmse``, ``capacity``) and
        ``linear_baseline_nmse``: the same readout fitted directly on the raw
        input windows, i.e. what we get with no reservoir at all. The reservoir
        is only earning its keep if it beats that.
    """
    protocol = build_protocol(seed=seed, **protocol_kwargs)
    (X_train, y_train), (X_test, y_test) = NARMA10Task(length=length, window_length=window_length, seed=seed).generate()

    readout = RegressionReadout(alpha=alpha)
    readout.fit(protocol.run(X_train), y_train)
    metrics, _ = readout.evaluate(protocol.run(X_test), y_test)

    baseline_readout = RegressionReadout(alpha=alpha)
    baseline_readout.fit(X_train[:, :, 0], y_train)
    baseline, _ = baseline_readout.evaluate(X_test[:, :, 0], y_test)
    metrics["linear_baseline_nmse"] = baseline["nmse"]
    return metrics


def _print_table(title, param, results, columns):
    print(f"\n{title}")
    print(f"{param:>6}  " + "  ".join(f"{c:>8}" for c in columns))
    for key, m in results.items():
        if key != "total_capacity":
            print(f"{key:>6}  " + "  ".join(f"{m[c]:>8.3f}" for c in columns))
    print(f"total capacity: {results['total_capacity']:.3f}")


if __name__ == "__main__":
    _print_table("Short-term memory", "delay", run_memory_benchmark(), ["nmse", "capacity"])
    _print_table("Temporal parity", "order", run_parity_benchmark(), ["nmse", "capacity", "accuracy"])

    narma = run_narma_benchmark()
    print("\nNARMA10")
    print(f"nmse: {narma['nmse']:.3f}  (linear baseline on raw inputs: {narma['linear_baseline_nmse']:.3f})")
