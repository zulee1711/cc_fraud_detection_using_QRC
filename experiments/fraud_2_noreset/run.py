"""fraud_2 without input reset: how fast the sliding-window QRC degrades with window length when the input qubit is
not reset between encoding steps.

Everything else is fraud_2 (experiments/fraud_2/run.py): data/sc3_c1000_x6, feature set 1, 1 input + 3 memory
qubits, numpy exact simulation, window lengths 1-30, reservoir seeds 42-51, the same readout and threshold. Two
reservoirs at their CLI defaults: random circuit (depth 4) and TFIM (v = 1, t = 1, 2 Trotter steps). Readout:
XYZ-strings (45 observables, 3 settings), exact and at 10^5 / 10^4 shots per window, all from one reservoir run.

The reset counterparts, logistic, ESN-45 and LSTM are the fraud_2 records in results/fraud_2/runs; aggregate.py in
results/fraud_2_noreset puts both side by side.

Commands (from the repository root):
    python -m experiments.fraud_2_noreset.run qrc --reservoir {random,tfim} --window-length L --seed S
    python -m experiments.fraud_2_noreset.run jobs     write jobs.txt next to this script
    python -m experiments.fraud_2_noreset.run check    exact features == NumpyBackend(reset_inputs=False)
experiments/fraud_2_noreset/run_jobs.sh runs every line of jobs.txt.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator

from experiments import cli
from experiments.cli import PROJECT_ROOT
from experiments.fraud_2.run import (BUDGETS, DATA, RESERVOIR_ARGS, SEEDS, WINDOW_LENGTHS, basis_change, evaluate,
                                     pauli_sets, settings_of, shot_features)
from qrc import pipeline
from qrc.backends.numpy_backend import NumpyBackend, _encoding_gates

RESULTS = PROJECT_ROOT / "results" / "fraud_2_noreset"
JOBS = Path(__file__).with_name("jobs.txt")
RESERVOIRS = ("random", "tfim")
SETS = ("XYZ-strings",)
OBSERVABLES = "XYZ-strings"  # the CLI run's observables (checked against the exact variant)


class NoResetSettingsBackend(NumpyBackend):
    """NumpyBackend without input reset (pure statevectors) that also keeps the features of every budget.

    `variants` collects one dict {(set, budget): (M, n_obs) float32} per run_batch call, in pipeline.SPLITS order."""

    def __init__(self, seed):
        super().__init__(reset_inputs=False)
        self.seed = seed
        self.variants = []

    def run_batch(self, windows, encoder, reservoir, observables):
        n = reservoir.num_qubits
        paulis = pauli_sets(n)
        settings = {s: settings_of(paulis[s]) for s in SETS}
        bases = sorted({b for st in settings.values() for b, _ in st})
        input_width = encoder.input_width(windows.shape[2])
        gates = _encoding_gates(encoder, input_width)
        values = windows[..., np.arange(input_width) % windows.shape[2]].astype(float)
        U = Operator(reservoir.circuit()).data
        P = observables.to_matrix(array=True)
        V = {b: basis_change(b) for b in bases}
        rng = np.random.default_rng((self.seed, len(self.variants)))
        D = U.shape[0]
        chunk = max(1, self.max_chunk_bytes // (16 * D * D * 2))

        exact, out = [], {(s, b): [] for s in SETS for b in BUDGETS}
        for start in range(0, len(values), chunk):
            psi = self._statevectors(values[start:start + chunk], gates, U, encoder.num_qubits)
            exact.append(np.einsum("ma,pab,mb->mp", psi.conj(), P, psi).real)
            probs = {b: np.abs(psi @ V[b].T) ** 2 for b in bases}
            probs = {b: p / p.sum(1, keepdims=True) for b, p in probs.items()}
            for s in SETS:
                for budget in BUDGETS:
                    per_setting = None if budget is None else budget // len(settings[s])
                    out[(s, budget)].append(shot_features(probs, settings[s], n, per_setting, rng)
                                            .astype(np.float32))
        self.variants.append({k: np.concatenate(v) for k, v in out.items()})
        return np.concatenate(exact)


def qrc_args(reservoir, L, seed):
    argv = ["qrc", *RESERVOIR_ARGS[reservoir], *DATA, "--window-length", str(L), "--n-mem-qubits", "3",
            "--observables", OBSERVABLES, "--backend", "numpy", "--seed", str(seed)]
    args = cli.build_parser().parse_args(argv)
    assert not args.reset_inputs
    return args


def run_qrc(reservoir, L, seed, results=RESULTS):
    name = f"qrc_{reservoir}_noreset_L{L}_s{seed}"
    path = results / "runs" / f"{name}.json"
    if path.exists():
        print(f"{name}: exists, skipped")
        return
    args = qrc_args(reservoir, L, seed)
    data = pipeline.prepare_data(args)
    backend = NoResetSettingsBackend(seed)
    t0 = time.perf_counter()
    exact, info = pipeline.qrc_features(data, args, backend=backend)
    seconds = time.perf_counter() - t0
    records = []
    for (s, budget) in backend.variants[0]:
        features = dict(zip(pipeline.SPLITS, (v[(s, budget)] for v in backend.variants)))
        if s == OBSERVABLES and budget is None:  # same observables as the CLI run, any order
            assert np.allclose(np.sort(features["test"][:200], 1), np.sort(exact["test"][:200], 1), atol=1e-5)
        t1 = time.perf_counter()
        metrics, _ = evaluate(features, data, args)
        records.append(dict(model="qrc", reservoir=reservoir, reset_inputs=False, L=L, seed=seed, observables=s,
                            budget=budget, n_features=features["train"].shape[1], reservoir_seconds=seconds,
                            readout_seconds=time.perf_counter() - t1, metrics=metrics, **info))
        print(f"{name} {s} {budget or 'exact'}: customer AP val "
              f"{metrics['validation']['average_precision_customer']:.3f}", flush=True)
    path.write_text(json.dumps(records, indent=1))


def check():
    """Exact features from the stored distributions vs NumpyBackend(reset_inputs=False), both reservoirs."""
    data = pipeline.prepare_data(qrc_args("random", 5, 42))
    small = {f"X_{s}": data["X_test"][:300] for s in pipeline.SPLITS}
    paulis = pipeline.build_observables(OBSERVABLES, 4)
    grouped = [p for _, g in settings_of(paulis) for p in g]
    order = [list(paulis).index(p) for p in grouped]
    for reservoir in RESERVOIRS:
        args = qrc_args(reservoir, 5, 42)
        backend = NoResetSettingsBackend(42)
        _, _ = pipeline.qrc_features(small, args, backend=backend)
        reference, _ = pipeline.qrc_features(small, args, backend=NumpyBackend(reset_inputs=False))
        ours = backend.variants[0][(OBSERVABLES, None)]
        err = np.abs(ours - reference["test"][:, order]).max()
        assert err < 1e-5, f"{reservoir}: {err:.2e}"
        noisy = np.abs(backend.variants[0][(OBSERVABLES, 10_000)] - ours).mean()
        print(f"{reservoir}: exact match ({err:.1e}); 10⁴-shot feature error {noisy:.4f}")


def jobs(path=JOBS):
    lines = [f"qrc --reservoir {r} --window-length {L} --seed {s}"
             for L in sorted(WINDOW_LENGTHS, reverse=True) for r in RESERVOIRS for s in SEEDS]
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("qrc", "jobs", "check"))
    parser.add_argument("--reservoir", choices=RESERVOIRS)
    parser.add_argument("--window-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    a = parser.parse_args()
    if a.command == "qrc":
        (a.results_dir / "runs").mkdir(parents=True, exist_ok=True)
        run_qrc(a.reservoir, a.window_length, a.seed, a.results_dir)
    elif a.command == "jobs":
        jobs()
    else:
        check()
