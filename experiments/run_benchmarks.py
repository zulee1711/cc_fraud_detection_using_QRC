"""Run benchmark tasks through the QRC pipeline.

Examples:
    python -m experiments.run_benchmarks --tasks ipc
    python -m experiments.run_benchmarks --tasks ipc --hamiltonian tfim
    python -m experiments.run_benchmarks --tasks ipc --hamiltonian native-exchange
    python -m experiments.run_benchmarks --tasks ipc --hamiltonian random
    python -m experiments.run_benchmarks --tasks ipc --backend shots --precision 0.01

The selected reservoir is either a random circuit or a Hamiltonian reservoir.
The exact estimator is the reference; gaussian and shots provide finite-
precision expectation-value estimates.
"""

import argparse
import json

import numpy as np

from experiments.cli import (
    BACKEND_NAMES,
    HAMILTONIAN_BUILDERS,
    make_backend,
)
from qrc.backends.spinpulse import DD_MODES, NOISE_TYPES, PULSE_SHAPES, SpinPulseConfig
from qrc.benchmarks import IPCTask, MemoryTask, NARMA10Task, ParityTask
from qrc.encodings import AngleEncoding
from qrc.observables import build_observables
from qrc.protocol import QRCProtocol
from qrc.readout import RegressionReadout
from qrc.reservoirs import HamiltonianReservoir, RandomCircuitReservoir


def build_protocol(
    n_input_qubits=1,
    n_mem_qubits=3,
    depth=2,
    observables_spec="Z+ZZ",
    reset_inputs=True,
    backend="estimator-exact",
    precision=0.05,
    seed=42,
    hamiltonian="random",
    steps=2,
    spinpulse=None,
):
    """Build the encoder -> reservoir -> observables protocol."""
    encoder = AngleEncoding(num_qubits=n_input_qubits)
    num_qubits = n_input_qubits + n_mem_qubits

    if hamiltonian == "random":
        reservoir = RandomCircuitReservoir(
            num_input_qubits=n_input_qubits,
            num_mem_qubits=n_mem_qubits,
            depth=depth,
            rng=seed,
        )
    else:
        if hamiltonian not in HAMILTONIAN_BUILDERS:
            valid = [*HAMILTONIAN_BUILDERS.keys()]
            raise ValueError(
                f"Unknown Hamiltonian {hamiltonian!r}. " f"Choose from: {valid}"
            )

        hamiltonian_operator = HAMILTONIAN_BUILDERS[hamiltonian](num_qubits, rng=seed)

        reservoir = HamiltonianReservoir(
            hamiltonian=hamiltonian_operator,
            num_input_qubits=n_input_qubits,
            time=1.0,
            steps=steps,
        )

    observables = build_observables(
        observables_spec,
        reservoir.num_qubits,
    )

    return QRCProtocol(
        encoder,
        reservoir,
        observables,
        make_backend(
            backend,
            precision,
            seed,
            reset_inputs=reset_inputs,
            spinpulse=spinpulse,
        ),
    )


def _score_sweep(protocol, tasks, alpha):
    """Run the reservoir once, then fit a readout for each task."""
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


def run_memory_benchmark(
    window_length=10,
    length=1000,
    seed=42,
    alpha=1e-6,
    **protocol_kwargs,
):
    """Measure memory capacity at each input delay."""
    protocol = build_protocol(seed=seed, **protocol_kwargs)

    tasks = [
        MemoryTask(
            delay=delay,
            length=length,
            window_length=window_length,
            seed=seed,
        )
        for delay in range(window_length)
    ]

    results = {
        task.delay: metrics
        for task, (metrics, _, _) in zip(tasks, _score_sweep(protocol, tasks, alpha))
    }

    results["total_capacity"] = sum(
        results[delay]["capacity"] for delay in range(window_length)
    )

    return results


def run_parity_benchmark(
    window_length=10,
    length=1000,
    seed=42,
    alpha=1e-6,
    **protocol_kwargs,
):
    """Measure the capacity to reconstruct temporal parity targets."""
    protocol = build_protocol(seed=seed, **protocol_kwargs)

    tasks = [
        ParityTask(
            order=order,
            length=length,
            window_length=window_length,
            seed=seed,
        )
        for order in range(1, window_length + 1)
    ]

    results = {}

    for task, (metrics, y_test, y_pred) in zip(
        tasks, _score_sweep(protocol, tasks, alpha)
    ):
        metrics["accuracy"] = float(np.mean((y_pred >= 0.5) == y_test))
        results[task.order] = metrics

    results["total_capacity"] = sum(
        results[order]["capacity"] for order in range(1, window_length + 1)
    )

    return results


def run_narma_benchmark(
    window_length=10,
    length=2000,
    seed=42,
    alpha=1e-6,
    **protocol_kwargs,
):
    """Evaluate NARMA10 prediction and compare against raw inputs."""
    protocol = build_protocol(seed=seed, **protocol_kwargs)

    (X_train, y_train), (X_test, y_test) = NARMA10Task(
        length=length,
        window_length=window_length,
        seed=seed,
    ).generate()

    readout = RegressionReadout(alpha=alpha)
    readout.fit(protocol.run(X_train), y_train)

    metrics, _ = readout.evaluate(protocol.run(X_test), y_test)

    baseline_readout = RegressionReadout(alpha=alpha)
    baseline_readout.fit(X_train[:, :, 0], y_train)

    baseline, _ = baseline_readout.evaluate(
        X_test[:, :, 0],
        y_test,
    )

    metrics["linear_baseline_nmse"] = baseline["nmse"]

    return metrics


def run_ipc_benchmark(
    length=3000,
    window_length=15,
    max_degree=3,
    max_delay=10,
    washout=200,
    test_split=0.2,
    threshold=1e-3,
    seed=42,
    alpha=1e-6,
    **protocol_kwargs,
):
    """Measure degree-wise information processing capacity."""
    task = IPCTask(
        length=length,
        window_length=window_length,
        max_degree=max_degree,
        max_delay=max_delay,
        washout=washout,
        test_split=test_split,
        threshold=threshold,
        seed=seed,
    )

    (X_train, Y_train), (X_test, Y_test) = task.generate()

    protocol = build_protocol(seed=seed, **protocol_kwargs)

    readout = RegressionReadout(alpha=alpha)
    readout.fit(protocol.run(X_train), Y_train)

    Y_pred = readout.predict(protocol.run(X_test))

    return task.compute_capacities(
        Y_pred, Y_test, n_observables=len(protocol.observables)
    )


def _print_table(title, param, results, columns):
    print(f"\n{title}")
    print(f"{param:>6}  " + "  ".join(f"{column:>8}" for column in columns))

    for key, metrics in results.items():
        if key == "total_capacity":
            continue

        print(
            f"{key:>6}  " + "  ".join(f"{metrics[column]:>8.3f}" for column in columns)
        )

    if "total_capacity" in results:
        print(f"total capacity: {results['total_capacity']:.3f}")


def build_spinpulse_config(args):
    """Build generic or realistic SpinPulse hardware/noise settings."""
    common = dict(
        noise_type=args.noise,
        only_idle=args.only_idle,
        dynamical_decoupling=args.dd,
        pulse_shape=args.pulse_shape,
        noise_samples=args.noise_samples,
        precision=args.precision,
        seed=args.seed,
    )
    if args.hardware == "realistic":
        return SpinPulseConfig.realistic(
            step_ns=args.time_step_ns,
            t2_us=args.t2_us,
            tj_us=args.tj_us or None,
            **common,
        )
    return SpinPulseConfig(T2S=args.t2, TJS=args.tj, **common)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("memory", "parity", "narma", "ipc"),
        default=["memory", "parity", "narma", "ipc"],
        help="benchmarks to run; defaults to all",
    )

    parser.add_argument(
        "--backend",
        nargs="+",
        choices=BACKEND_NAMES,
        default=list(BACKEND_NAMES),
        help="backend(s) to compare",
    )

    parser.add_argument(
        "--hamiltonian",
        choices=("random", *HAMILTONIAN_BUILDERS.keys()),
        default="native-exchange",
        help="reservoir model; random selects a random-circuit reservoir",
    )

    parser.add_argument(
        "--precision",
        type=float,
        default=0.05,
        help="target standard error for finite-precision backends",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1e-6)

    # IPC options
    parser.add_argument("--ipc-length", type=int, default=3000)
    parser.add_argument("--ipc-window-length", type=int, default=15)
    parser.add_argument(
        "--ipc-reservoir-depth",
        type=int,
        default=2,
        help="random-circuit depth or Hamiltonian Trotter steps for IPC",
    )
    parser.add_argument(
        "--observables",
        dest="observables_spec",
        default="Z+ZZ",
        help="observable specification used by the reservoir protocol (for example, Z+ZZ)",
    )
    parser.add_argument("--max-degree", type=int, default=3)
    parser.add_argument("--max-delay", type=int, default=10)
    parser.add_argument("--washout", type=int, default=200)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=1e-3)

    # SpinPulse hardware/noise options (used only with --backend spinpulse)
    spin = parser.add_argument_group(
        "SpinPulse backend (only with --backend spinpulse)"
    )
    spin.add_argument(
        "--hardware",
        choices=("generic", "realistic"),
        default="generic",
        help="generic SpinPulse settings or realistic spin-qubit hardware parameters",
    )
    spin.add_argument(
        "--noise",
        choices=NOISE_TYPES,
        default="pink",
        help="qubit-frequency noise model: none, white, pink (1/f), or quasistatic",
    )
    spin.add_argument(
        "--t2",
        type=float,
        default=100.0,
        help="generic hardware T2* in SpinPulse time units",
    )
    spin.add_argument(
        "--tj",
        type=float,
        default=None,
        help="generic hardware exchange-coupling dephasing time",
    )
    spin.add_argument(
        "--time-step-ns",
        type=float,
        default=10.0,
        help="realistic hardware simulation time step in ns",
    )
    spin.add_argument(
        "--t2-us",
        type=float,
        default=10.0,
        help="realistic hardware qubit T2* in microseconds",
    )
    spin.add_argument(
        "--tj-us",
        type=float,
        default=5.0,
        help="realistic hardware coupling dephasing time in microseconds; 0 disables it",
    )
    spin.add_argument(
        "--only-idle",
        action="store_true",
        help="apply frequency noise to idle qubits only",
    )
    spin.add_argument(
        "--dd", choices=DD_MODES, default="none", help="dynamical decoupling mode"
    )
    spin.add_argument(
        "--pulse-shape",
        choices=PULSE_SHAPES,
        default="gaussian",
        help="control pulse envelope",
    )
    spin.add_argument(
        "--noise-samples",
        type=int,
        default=20,
        help="noise realizations averaged per window",
    )

    args = parser.parse_args()
    if args.ipc_reservoir_depth < 1:
        parser.error("--ipc-reservoir-depth must be at least 1")

    for backend in args.backend:
        kwargs = {
            "backend": backend,
            "precision": args.precision,
            "seed": args.seed,
            "hamiltonian": args.hamiltonian,
            "observables_spec": args.observables_spec,
            "spinpulse": (
                build_spinpulse_config(args) if backend == "spinpulse" else None
            ),
        }

        print(
            f"\n=== Backend: {backend}"
            + ("" if backend == "estimator-exact" else f" (precision {args.precision})")
        )
        print(f"Reservoir: {args.hamiltonian}")

        if "memory" in args.tasks:
            _print_table(
                "Short-term memory",
                "delay",
                run_memory_benchmark(alpha=args.alpha, **kwargs),
                ["nmse", "capacity"],
            )

        if "parity" in args.tasks:
            _print_table(
                "Temporal parity",
                "order",
                run_parity_benchmark(alpha=args.alpha, **kwargs),
                ["nmse", "capacity", "accuracy"],
            )

        if "narma" in args.tasks:
            narma = run_narma_benchmark(
                alpha=args.alpha,
                **kwargs,
            )

            print("\nNARMA10")
            print(
                f"nmse: {narma['nmse']:.3f} "
                f"(linear baseline on raw inputs: "
                f"{narma['linear_baseline_nmse']:.3f})"
            )

        if "ipc" in args.tasks:
            reservoir_depth = (
                {"depth": args.ipc_reservoir_depth}
                if args.hamiltonian == "random"
                else {"steps": args.ipc_reservoir_depth}
            )
            ipc = run_ipc_benchmark(
                length=args.ipc_length,
                window_length=args.ipc_window_length,
                max_degree=args.max_degree,
                max_delay=args.max_delay,
                washout=args.washout,
                test_split=args.test_split,
                threshold=args.threshold,
                alpha=args.alpha,
                **reservoir_depth,
                **kwargs,
            )

            print("\nInformation Processing Capacity (IPC)")
            print(json.dumps(ipc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
