"""E6: how does hardware noise affect the reservoir, and can it help? SpinPulse vs depolarizing.

Two parts, both on the short-term memory task of experiments.spinpulse_memory (TFIM, 1 input
qubit with reset, 2 Trotter steps):

calibrate
    For each SpinPulse noise type (pink, quasistatic, white), find the qubit T2* (us, realistic
    chip, coupling noise off) at which RZZ(pi/2), with its echoes, has a target average gate
    fidelity. Noise types are then compared at equal two-qubit fidelity, not at equal T2*.
    Pure dephasing cannot give realistic 1q and 2q fidelities at once (a 1q pulse is 3x
    shorter), so the 1q fidelity that follows is reported too.

compare
    Memory capacity and feature signal of:
      ideal              the exact ideal circuit
      depolarizing       a depolarizing channel after every native gate at the same 2q fidelity
                         (1q infidelity = 2q / 5), the model of results/observable_sets/observables.py
      sp-<noise type>    SpinPulse at the calibrated T2*
    in exact mode and at the given shot precisions. Exact mode shows what noise does to the
    reservoir map itself (dephasing keeps populations, depolarizing shrinks everything);
    shots show what is left on hardware. A noisy model whose capacity beats the ideal one
    beyond the seed spread is a noise-induced advantage.

Which noise source matters at today's chip (qubit vs coupling noise) needs no new code:
    python -m experiments.spinpulse_memory --levels 1 --no-coupling-noise --label qubit-only
    python -m experiments.spinpulse_memory --levels 1 --no-qubit-noise --label coupling-only

    python -m experiments.spinpulse_noise_models calibrate                      (~10 min)
    python -m experiments.spinpulse_noise_models compare --noise-types pink --targets 0.99
                                                                    (~20 min; the full grid ~2.5 h)
Ideal and depolarizing runs take seconds; each SpinPulse run about 0.3 s per window with the
default 20 noise samples. compare appends to --out, so it can be run in pieces.
"""

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from experiments.cli import PROJECT_ROOT
from experiments.spinpulse_calibration import gate_report
from qrc.backends.spinpulse import SpinPulseBackend, SpinPulseConfig, _sample_diagonal_expectations
from qrc.benchmarks import MemoryTask
from qrc.encodings import AngleEncoding
from qrc.hamiltonians import tfim
from qrc.observables import build_observables
from qrc.protocol import QRCProtocol
from qrc.readout import RegressionReadout
from qrc.reservoirs import HamiltonianReservoir

OUT_DIR = PROJECT_ROOT / "results" / "spinpulse_noise_models"
CALIBRATION_FIELDS = ["noise", "target_f2", "t2_us", "f2", "f1", "samples"]
COMPARE_FIELDS = [
    "model", "noise", "target_f2", "t2_us", "seed", "precision", "n_qubits", "window_length",
    "delay", "capacity", "signal", "seconds",
]
NOISE_SEED_OFFSET = 10_000
SHOT_SEED_OFFSET = 40_000


# the observable-set study's script; results/observable_sets is not tracked, so it is only there locally
DEPOLARIZING_SCRIPT = PROJECT_ROOT / "results" / "observable_sets" / "observables.py"


def depolarizing_model():
    """The density-matrix simulator of the observable-set study (P2), loaded from its script."""
    path = DEPOLARIZING_SCRIPT
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the depolarizing model needs the (untracked) observable-set study")
    spec = importlib.util.spec_from_file_location("observable_sets_p2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ calibrate


def calibrate_one(noise, target, args):
    """T2* (us) giving RZZ(pi/2) the target fidelity, by bisection on log T2*."""
    lo, hi = np.log(args.t2_min_us), np.log(args.t2_max_us)

    def fidelities(t2_us):
        config = SpinPulseConfig.realistic(step_ns=args.time_step_ns, t2_us=t2_us, tj_us=None,
                                           noise_type=noise, seed=args.seed)
        report = gate_report(SpinPulseBackend(config), samples=args.samples)
        return report["RZZ(pi/2)"][2], report["X (rx pi)"][2]

    for _ in range(args.iterations):
        mid = (lo + hi) / 2
        f2, _ = fidelities(np.exp(mid))
        lo, hi = (lo, mid) if f2 > target else (mid, hi)  # fidelity grows with T2*
    t2_us = float(np.exp((lo + hi) / 2))
    f2, f1 = fidelities(t2_us)
    return dict(noise=noise, target_f2=target, t2_us=round(t2_us, 4), f2=f2, f1=f1, samples=args.samples)


def calibrate(args):
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    with open(args.calibration, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CALIBRATION_FIELDS)
        writer.writeheader()
        for noise in args.noise_types:
            for target in args.targets:
                row = calibrate_one(noise, target, args)
                writer.writerow(row)
                f.flush()
                print(f"{noise:>12} target F2 {target}: T2* = {row['t2_us']:.3f} us -> "
                      f"F2 {row['f2']:.5f}, F1 {row['f1']:.6f}", flush=True)
    return args.calibration


# ------------------------------------------------------------------ compare


def exact_and_populations(rho, observables):
    """Exact Pauli expectations (M, K) and outcome probabilities (2^n, M) from density matrices."""
    ops = [SparsePauliOp(p).to_matrix() for p in observables]
    expectations = np.stack([np.einsum("ij,mji->m", op, rho).real for op in ops], axis=1)
    return expectations, np.real(np.diagonal(rho, axis1=1, axis2=2)).T


def depolarizing_features(model, X, reservoir, observables, f2, precision, rng):
    """Features of the P2 depolarizing model at 2q fidelity f2 (f2 = 1: the ideal circuit)."""
    n = reservoir.num_qubits
    noise = (1 - f2, (1 - f2) / 5, 0.0)  # (2q infidelity, 1q infidelity, readout flip): no readout error
    rho = model.density_matrices(X, model.reservoir_ops(reservoir, n), n, noise)
    expectations, populations = exact_and_populations(rho, observables)
    if precision == 0:
        return expectations
    return _sample_diagonal_expectations(populations, observables, int(np.ceil(1 / precision**2)), rng)


def capacity_rows(train, test, tasks, alpha, **common):
    signal = float(np.std(train, axis=0).mean())
    rows = []
    for task in tasks:
        (_, y_train), (_, y_test) = task.generate()
        readout = RegressionReadout(alpha=alpha)
        readout.fit(train, y_train)
        metrics, _ = readout.evaluate(test, y_test)
        rows.append(dict(common, delay=task.delay, capacity=metrics["capacity"], signal=signal))
    return rows


def compare(args):
    calibration = {}
    if args.calibration.exists():
        with open(args.calibration, newline="") as f:
            for r in csv.DictReader(f):
                calibration[(r["noise"], float(r["target_f2"]))] = float(r["t2_us"])
    missing = [(n, t) for n in args.noise_types for t in args.targets if (n, t) not in calibration]
    if missing:
        raise SystemExit(f"no calibration for {missing}: run `calibrate` first (or pass --calibration)")

    model = depolarizing_model()
    n_qubits = 1 + args.n_mem_qubits
    observables = build_observables(args.observables, n_qubits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARE_FIELDS)
        if write_header:
            writer.writeheader()
        for seed in args.seeds:
            reservoir = HamiltonianReservoir(tfim(n_qubits, rng=seed), 1, time=1.0, steps=args.trotter_steps)
            tasks = [MemoryTask(delay=d, length=args.length, window_length=args.window_length, seed=seed)
                     for d in range(args.window_length)]
            (X_train, _), (X_test, _) = tasks[0].generate()
            for precision in args.precisions:
                runs = [("ideal", "", 1.0, None)]
                runs += [("depolarizing", "", t, None) for t in args.targets]
                runs += [(f"sp-{n}", n, t, calibration[(n, t)]) for n in args.noise_types for t in args.targets]
                for name, noise, target, t2_us in runs:
                    start = time.perf_counter()
                    if t2_us is None:  # ideal and depolarizing: the P2 density-matrix simulator
                        rng = np.random.default_rng([seed + SHOT_SEED_OFFSET, round(precision * 1e6)])
                        train, test = (depolarizing_features(model, X, reservoir, observables, target, precision, rng)
                                       for X in (X_train, X_test))
                    else:
                        config = SpinPulseConfig.realistic(
                            step_ns=args.time_step_ns, t2_us=t2_us, tj_us=None, noise_type=noise,
                            noise_samples=args.noise_samples, precision=precision,
                            seed=seed + NOISE_SEED_OFFSET,
                        )
                        protocol = QRCProtocol(AngleEncoding(1), reservoir, observables,
                                               SpinPulseBackend(config, reset_inputs=True))
                        train, test = protocol.run(X_train), protocol.run(X_test)
                    rows = capacity_rows(
                        train, test, tasks, args.alpha, model=name, noise=noise,
                        target_f2=target if name != "ideal" else "", t2_us=t2_us or "", seed=seed,
                        precision=precision, n_qubits=n_qubits, window_length=args.window_length,
                        seconds=round(time.perf_counter() - start, 2),
                    )
                    writer.writerows(rows)
                    f.flush()
                    total = sum(r["capacity"] for r in rows)
                    print(f"seed {seed} precision {precision:g} {name:>16} F2 {target}: "
                          f"total MC {total:.3f}, signal {rows[0]['signal']:.4f}", flush=True)
    return args.out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=("calibrate", "compare"))
    p.add_argument("--noise-types", nargs="+", default=["pink", "quasistatic", "white"])
    p.add_argument("--targets", nargs="+", type=float, default=[0.995, 0.99, 0.98],
                   help="two-qubit (RZZ) average gate fidelities to compare at")
    p.add_argument("--time-step-ns", type=float, default=10.0)
    p.add_argument("--calibration", type=Path, default=OUT_DIR / "calibration.csv")
    # calibrate
    p.add_argument("--samples", type=int, default=100, help="calibrate: noise samples per gate fidelity")
    p.add_argument("--iterations", type=int, default=12, help="calibrate: bisection steps on log T2*")
    p.add_argument("--t2-min-us", type=float, default=0.1)
    p.add_argument("--t2-max-us", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=0, help="calibrate: noise seed")
    # compare
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--precisions", nargs="+", type=float, default=[0.0, 0.05])
    p.add_argument("--noise-samples", type=int, default=20)
    p.add_argument("--n-mem-qubits", type=int, default=3)
    p.add_argument("--trotter-steps", type=int, default=2)
    p.add_argument("--observables", default="Z+ZZ")
    p.add_argument("--window-length", type=int, default=10)
    p.add_argument("--length", type=int, default=500)
    p.add_argument("--alpha", type=float, default=1e-6)
    p.add_argument("--out", type=Path, default=OUT_DIR / "compare.csv")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return calibrate(args) if args.part == "calibrate" else compare(args)


if __name__ == "__main__":
    main()