"""
Short-term memory capacity of the reservoir on SpinPulse hardware.

Runs the memory benchmark (``qrc/benchmarks/memory.py``: recall the input from ``d``
steps ago, d = 0 .. L-1) for a Hamiltonian reservoir with 1 input qubit and input reset,
at several hardware levels, and writes one CSV row per (level, seed, delay).

Hardware levels:
    ideal       exact ideal circuit (NumpyBackend), the reference
    sp-ideal    SpinPulse realistic chip without noise (its intrinsic gate error only)
    <factor>    SpinPulse realistic chip with T2* = 10 us x factor and TJ* = 5 us x factor
                (1 = today's devices, 10 = ten times better coherence)

Reservoir seed and noise seed are separate: the reservoir and the input series use
``seed``, the noise traces use ``seed + 10000``. Rows already in the output file are
skipped, so an interrupted sweep resumes where it stopped.

    python -m experiments.spinpulse_memory --levels ideal sp-ideal 1 2 5 10 30
    python -m experiments.spinpulse_memory --levels 2 --no-coupling-noise --label qubit-noise-only
    python -m experiments.spinpulse_memory --levels ideal 1 2 5 10 30 --precision 0.05
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments.cli import PROJECT_ROOT
from qrc.pipeline import HAMILTONIAN_BUILDERS
from qrc.backends.numpy_backend import NumpyBackend
from qrc.backends.spinpulse import NOISE_TYPES, SpinPulseBackend, SpinPulseConfig
from qrc.benchmarks import MemoryTask
from qrc.encodings import AngleEncoding
from qrc.hamiltonians import HamiltonianTopology
from qrc.observables import build_observables
from qrc.protocol import QRCProtocol
from qrc.readout import RegressionReadout
from qrc.reservoirs import HamiltonianReservoir

FIELDS = [
    "label", "level", "t2_us", "tj_us", "noise", "precision", "noise_samples", "reset",
    "hamiltonian", "trotter_steps", "n_qubits", "window_length", "seed",
    "delay", "capacity", "nmse", "signal", "noise_stderr", "window_us", "seconds",
]
NOISE_SEED_OFFSET = 10_000


def make_backend(level, args, seed):
    """Backend for one hardware level, and the (T2*, TJ*) it uses in us (None = off)."""
    if level == "ideal":
        return NumpyBackend(reset_inputs=args.reset), None, None
    common = dict(
        step_ns=args.time_step_ns,
        noise_samples=args.noise_samples,
        precision=args.precision,
        seed=seed + NOISE_SEED_OFFSET,
    )
    if level == "sp-ideal":
        config = SpinPulseConfig.realistic(noise_type="none", **common)
        return SpinPulseBackend(config, reset_inputs=args.reset), None, None
    factor = float(level)
    t2_us = 10.0 * factor if args.qubit_noise else 1e9  # 1e9 us: no qubit dephasing
    tj_us = 5.0 * factor if args.coupling_noise else None
    config = SpinPulseConfig.realistic(t2_us=t2_us, tj_us=tj_us, noise_type=args.noise, **common)
    return SpinPulseBackend(config, reset_inputs=args.reset), t2_us, tj_us


def run_one(level, seed, args):
    """Memory capacity at every delay for one hardware level and seed: list of CSV rows."""
    n_qubits = 1 + args.n_mem_qubits
    hamiltonian = HAMILTONIAN_BUILDERS[args.hamiltonian](
        n_qubits, topology=HamiltonianTopology.LINEAR, rng=seed
    )
    reservoir = HamiltonianReservoir(hamiltonian, 1, time=args.evolution_time, steps=args.trotter_steps)
    observables = build_observables(args.observables, n_qubits)
    backend, t2_us, tj_us = make_backend(level, args, seed)
    protocol = QRCProtocol(AngleEncoding(num_qubits=1), reservoir, observables, backend)
    tasks = [
        MemoryTask(delay=d, length=args.length, window_length=args.window_length, seed=seed)
        for d in range(args.window_length)
    ]

    # Run the reservoir once on the shared windows, then fit one readout per delay
    # (as run_benchmarks._score_sweep, which does not expose the features).
    start = time.perf_counter()
    (X_train, _), (X_test, _) = tasks[0].generate()
    train_outputs = protocol.run(X_train)
    stderr = getattr(backend, "last_stderr", None)
    test_outputs = protocol.run(X_test)
    seconds = time.perf_counter() - start
    scores = []
    for task in tasks:
        (_, y_train), (_, y_test) = task.generate()
        readout = RegressionReadout(alpha=args.alpha)
        readout.fit(train_outputs, y_train)
        metrics, _ = readout.evaluate(test_outputs, y_test)
        scores.append((metrics, None, None))
    # Signal: how much the features vary between windows (noise shrinks it; with shots,
    # memory survives only while it stays above the shot noise ~precision).
    signal = float(np.std(train_outputs, axis=0).mean())
    noise_stderr = "" if stderr is None else float(np.mean(stderr))

    window_us = ""
    if isinstance(backend, SpinPulseBackend):
        steps = args.window_length * backend._timeline.step_duration
        window_us = round(steps * backend.config.time_step_ns / 1000, 3)
    return [
        dict(
            label=args.label, level=level, t2_us=t2_us or "", tj_us=tj_us or "",
            noise="" if level in ("ideal", "sp-ideal") else args.noise,
            precision=args.precision if level != "ideal" else 0.0,
            noise_samples=args.noise_samples, reset=args.reset,
            hamiltonian=args.hamiltonian, trotter_steps=args.trotter_steps, n_qubits=n_qubits,
            window_length=args.window_length, seed=seed, delay=task.delay,
            capacity=metrics["capacity"], nmse=metrics["nmse"],
            signal=signal, noise_stderr=noise_stderr,
            window_us=window_us, seconds=round(seconds, 2),
        )
        for task, (metrics, _, _) in zip(tasks, scores)
    ]


def done_runs(path, label):
    """(level, seed) pairs already in the output file for this label."""
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["level"], int(r["seed"])) for r in csv.DictReader(f) if r["label"] == label}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+", default=["ideal", "sp-ideal", "1", "2", "5", "10", "30"],
                   help="hardware levels: ideal, sp-ideal, or coherence factors (1 = today's chip)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    p.add_argument("--noise", choices=[n for n in NOISE_TYPES if n != "none"], default="pink")
    p.add_argument("--qubit-noise", action=argparse.BooleanOptionalAction, default=True,
                   help="qubit dephasing (T2*); --no-qubit-noise for coupling noise only")
    p.add_argument("--coupling-noise", action=argparse.BooleanOptionalAction, default=True,
                   help="exchange-coupling noise (TJ*); --no-coupling-noise for qubit noise only")
    p.add_argument("--precision", type=float, default=0.0, help="0 = exact; p -> ceil(1/p^2) shots")
    p.add_argument("--noise-samples", type=int, default=20, help="noise realizations per window")
    p.add_argument("--reset", action=argparse.BooleanOptionalAction, default=True,
                   help="reset the input qubit before each step (exact for small registers)")
    p.add_argument("--hamiltonian", choices=tuple(HAMILTONIAN_BUILDERS), default="tfim")
    p.add_argument("--trotter-steps", type=int, default=2)
    p.add_argument("--evolution-time", type=float, default=1.0)
    p.add_argument("--n-mem-qubits", type=int, default=3)
    p.add_argument("--observables", default="Z+ZZ")
    p.add_argument("--window-length", type=int, default=10)
    p.add_argument("--length", type=int, default=1000, help="time steps of the input series")
    p.add_argument("--alpha", type=float, default=1e-6, help="ridge regularisation of the readout")
    p.add_argument("--time-step-ns", type=float, default=10.0)
    p.add_argument("--label", default="main", help="name of this sweep in the output file")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "spinpulse_memory" / "memory.csv")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_runs(args.out, args.label)
    write_header = not args.out.exists()

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for level in args.levels:
            for seed in args.seeds:
                if (level, seed) in done:
                    continue
                rows = run_one(level, seed, args)
                writer.writerows(rows)
                f.flush()  # keep finished runs if the sweep is interrupted
                total = sum(r["capacity"] for r in rows)
                window = f", window {rows[0]['window_us']} us" if rows[0]["window_us"] != "" else ""
                print(f"[{args.label}] level {level:>8}  seed {seed}: total MC {total:.3f}, "
                      f"signal {rows[0]['signal']:.4f}  ({rows[0]['seconds']} s{window})", flush=True)
    return args.out


if __name__ == "__main__":
    main()