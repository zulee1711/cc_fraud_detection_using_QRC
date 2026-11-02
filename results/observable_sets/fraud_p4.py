"""P4: fraud detection (arm B) with the observable sets of P2, under shot and gate noise (input reset on).

Data, windowing, encoding, reservoir, readout, decision threshold and metrics are the CLI's (experiments.cli), as in
results/fraud_sweep: data/sc3_c1000_x6, feature set 1 (log amount), L = 10, 1 input + 3 memory qubits, the CLI's
TFIM (`--hamiltonian tfim`, time 1, 2 Trotter steps, the P2/P3 reservoir), input reset. The backend is replaced by one that
simulates density matrices under the P2 noise model (observables.NOISE: depolarizing after every native gate,
reset preparation and readout flips, one noisy 1q gate per rotated qubit before readout) and keeps the outcome
distribution of every measurement setting. Each readout variant (observable set x shot budget) is derived from one
reservoir run per noise level, sampled per window and setting on train, validation and test alike.

For speed the noisy reservoir block is applied as one superoperator per step (built from the gate sequence by
observables.apply_ops on the D^2 basis matrices), the encoding as a batched unitary plus depolarizing on the input
qubit; --check compares this with the gate-by-gate simulation.

Usage (from the repository root, data in data/sc3_c1000_x6):
    python results/observable_sets/fraud_p4.py --check
    python results/observable_sets/fraud_p4.py --seed 42 --out results/observable_sets/data/p4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import observables as obs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, for experiments.cli
from experiments import cli  # noqa: E402
from qrc import pipeline  # noqa: E402
from qrc.backends.numpy_backend import NumpyBackend, _encoding_gates, _input_unitaries  # noqa: E402

DATA = ["--data", "load", "--data-dir", "data/sc3_c1000_x6", "--feature-set", "1"]
L = 10
LEVELS = ("ideal", "2q1", "2q2")
SETS = ("Z-w1", "Z-strings", "frames-3", "XYZ-strings", "XYZ+XYZ-2body")
BUDGETS = (1_000, 10_000, 100_000, None)
CHUNK = 20_000  # windows per simulation chunk


def qrc_args(seed, window_length=L):
    argv = ["qrc", "--hamiltonian", "tfim", *DATA, "--window-length", str(window_length), "--n-mem-qubits", "3",
            "--observables", "XYZ+XYZ-2body", "--backend", "numpy", "--seed", str(seed), "--reset-inputs"]
    return cli.build_parser().parse_args(argv)


def superoperator(ops, noise, n):
    """Matrix S with vec(channel(rho)) = S @ vec(rho), vec row-major, for the noisy gate sequence."""
    D = 2 ** n
    basis = np.eye(D * D, dtype=complex).reshape(D * D, D, D)
    return obs.apply_ops(basis, ops, noise, n).reshape(D * D, D * D).T


class NoisyBackend(NumpyBackend):
    """Density matrices under the P2 noise model; per run_batch call, appends the features of every (set, budget)
    variant to `variants` and returns the exact XYZ+XYZ-2body features (CLI observables)."""

    def __init__(self, level, seed, gate_by_gate=False):
        super().__init__(reset_inputs=True)
        self.noise = obs.NOISE[level]
        self.seed = seed
        self.gate_by_gate = gate_by_gate
        self.variants = []

    def states(self, x, encoder, ops, S, n):
        p2, p1, p_ro = self.noise
        gates = _encoding_gates(encoder, encoder.input_width(x.shape[2]))
        M, D = len(x), 2 ** n
        rho = np.zeros((M, D, D), complex)
        rho[:, 0, 0] = 1
        for t in range(x.shape[1]):
            if t > 0:
                v = rho.reshape(M, D // 2, 2, D // 2, 2)
                tr = v[:, :, 0, :, 0] + v[:, :, 1, :, 1]
                rho = np.zeros_like(rho)
                v = rho.reshape(M, D // 2, 2, D // 2, 2)
                v[:, :, 0, :, 0] = (1 - p_ro) * tr
                v[:, :, 1, :, 1] = p_ro * tr
            rho = obs.apply_gate(rho, _input_unitaries(x[:, t], gates, 1), (0,), n)
            rho = obs.depolarize(rho, (0,), obs.lam_of(p1, 1), n)
            if self.gate_by_gate:
                rho = obs.apply_ops(rho, ops, self.noise, n)
            else:
                rho = (rho.reshape(M, D * D) @ S.T).reshape(M, D, D)
        return rho

    def run_batch(self, windows, encoder, reservoir, observables):
        n = reservoir.num_qubits
        ops = obs.reservoir_ops(reservoir, n)
        S = None if self.gate_by_gate else superoperator(ops, self.noise, n)
        sets = {k: v for k, v in obs.pauli_settings(n, self.seed).items() if k in SETS}
        keys = {key: (V, rot) for s in sets.values() for key, V, rot, _ in s}
        P = observables.to_matrix(array=True)
        rng = np.random.default_rng((self.seed, len(self.variants)))
        input_width = encoder.input_width(windows.shape[2])
        values = windows[..., np.arange(input_width) % windows.shape[2]].astype(float)
        exact, out = [], {(s, b): [] for s in sets for b in BUDGETS}
        for start in range(0, len(values), CHUNK):
            rho = self.states(values[start:start + CHUNK], encoder, ops, S, n)
            exact.append(np.einsum("pab,mba->mp", P, rho).real)
            probs = {key: obs.outcome_probs(rho, V, rot, self.noise, n) for key, (V, rot) in keys.items()}
            for s, settings in sets.items():
                for budget in BUDGETS:
                    shots = None if budget is None else budget // len(settings)
                    out[(s, budget)].append(obs.features(probs, settings, n, shots, rng).astype(np.float32))
        self.variants.append({k: np.concatenate(v) for k, v in out.items()})
        return np.concatenate(exact)


def evaluate(features, data, args):
    """cli.run's evaluation: readout + threshold on validation, operational and per-category metrics."""
    metrics, scores = pipeline.fit_and_evaluate(features, data, args.alpha, args.min_recall)
    pipeline.add_derived_metrics(metrics, scores, data, args.top_k)
    return metrics


def flat(metrics):
    return {f"{split}_{k}": v for split, m in metrics.items() for k, v in m.items()}


def run(seed, levels, out):
    args = qrc_args(seed)
    data = pipeline.prepare_data(args)
    rows = []
    t0 = time.time()
    base = evaluate(pipeline.baseline_features(data), data, args)
    rows.append(dict(model="logistic", seed=seed, noise="", observables="raw window", n_settings=0, budget=np.nan,
                     n_features=data["X_train"].shape[1] * data["X_train"].shape[2], **flat(base)))
    print(f"logistic: customer AP val {base['validation']['average_precision_customer']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    for level in levels:
        t0 = time.time()
        backend = NoisyBackend(level, seed)
        pipeline.qrc_features(data, args, backend=backend)
        sim = time.time() - t0
        n_settings = {k: len(v) for k, v in obs.pauli_settings(4, seed).items()}
        for (s, budget) in backend.variants[0]:
            features = dict(zip(pipeline.SPLITS, (v[(s, budget)] for v in backend.variants)))
            m = evaluate(features, data, args)
            rows.append(dict(model="qrc", seed=seed, noise=level, observables=s, n_settings=n_settings[s],
                             budget=np.inf if budget is None else budget, n_features=features["train"].shape[1],
                             **flat(m)))
            print(f"{level} {s} {budget or 'exact'}: customer AP val "
                  f"{m['validation']['average_precision_customer']:.3f}", flush=True)
        print(f"{level}: simulation {sim:.0f}s, total {time.time() - t0:.0f}s", flush=True)
        del backend
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"fraud_seed{seed}.csv", index=False)


def check():
    """Ideal: exact CLI observables equal NumpyBackend. Noisy: superoperator path equals gate by gate."""
    args = qrc_args(42, window_length=5)
    data = pipeline.prepare_data(args)
    small = {f"X_{s}": data["X_test"][:200] for s in pipeline.SPLITS}
    ref = pipeline.build_protocol(args)[0]
    ref_features = ref.run(small["X_test"])  # NumpyBackend with reset (the CLI's backend)
    for level, gbg in (("ideal", False), ("2q2", False), ("2q2", True)):
        backend = NoisyBackend(level, 42, gate_by_gate=gbg)
        feats, _ = pipeline.qrc_features(small, args, backend=backend)
        if level == "ideal":
            err = np.abs(feats["test"] - ref_features).max()
            assert err < 1e-8, f"ideal features disagree with NumpyBackend: {err:.2e}"
        elif gbg:
            err = np.abs(feats["test"] - fast).max()
            assert err < 1e-10, f"superoperator path disagrees with gate-by-gate: {err:.2e}"
        else:
            fast = feats["test"]
        print(f"{level} {'gate-by-gate' if gbg else 'superoperator'}: ok", flush=True)
    print("check: ideal matches NumpyBackend, superoperator matches gate-by-gate", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--out", default="results/observable_sets/data/p4")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        run(args.seed, args.levels.split(","), Path(args.out))
