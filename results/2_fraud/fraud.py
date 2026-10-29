"""Report section 7.2: per-category AP vs window length for the sliding-window QRC on fraud detection, against
classical baselines on the same window.

QRC (CLI settings, experiments.cli): 1 input + 3 memory qubits, input reset, three reservoirs at their CLI defaults:
    random           random circuit, depth 4
    tfim             qrc.hamiltonians.tfim, v = 1, t = 1, 2 Trotter steps
    native-exchange  qrc.hamiltonians.native_exchange, v = 1, eta = 1, t = 1, 2 Trotter steps
Baselines: logistic regression on the raw window, echo state networks of each QRC readout width (7 and 45 units,
CLI defaults, untuned) and the LSTM (CLI defaults). Data: data/sc3_c1000_x6, feature set 1 (raw log amount).

Each QRC run goes through cli.qrc_features (CLI observables Z+ZZ) with a backend that also keeps, per window, the
outcome distribution of every measurement setting needed, so every readout variant comes from the same reservoir run:
    observables  Z+ZZ (7 obs., 1 setting) and XYZ-strings (45 obs., every all-X / all-Y / all-Z Pauli string, 3 settings)
    budget       exact, 10⁵ and 10⁴ shots per window, split evenly over the set's settings; per-setting multinomial
                 sampling of the exact distributions, train/validation/test all sampled (results/shots model)

Commands (from the repository root, PYTHONPATH=.):
    fraud.py qrc --reservoir {random,tfim,native-exchange} --window-length L --seed S [--save-scores]
    fraud.py baseline --window-length L
    fraud.py lstm --window-length L --seed S
    fraud.py esn --window-length L --seed S --n-units N
    fraud.py jobs     write jobs.txt (the full grid, most expensive first)
    fraud.py check    exact features from the stored distributions == NumpyBackend, every reservoir
Each run writes runs/<name>.json (one record per readout variant); existing records are skipped.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, PauliList

from experiments import cli
from qrc.backends.numpy_backend import NumpyBackend, _encoding_gates

HERE = Path("results/2_fraud")
RUNS = HERE / "runs"
DATA = ["--data", "load", "--data-dir", "data/sc3_c1000_x6", "--feature-set", "1"]
RESERVOIR_ARGS = {
    "random": ["--hamiltonian", "random", "--depth", "4"],
    "tfim": ["--hamiltonian", "tfim"],
    "native-exchange": ["--hamiltonian", "native-exchange"],
}
OBSERVABLES = "Z+ZZ"  # the CLI run's observables
SETS = ("Z+ZZ", "XYZ-strings")
BUDGETS = (None, 100_000, 10_000)
WINDOW_LENGTHS = (1, 3, 5, 10, 15, 20, 30)
SEEDS = tuple(range(42, 52))
ESN_UNITS = (7, 45)  # readout widths of the QRC on 4 qubits: Z+ZZ, XYZ-strings
SCORE_LENGTHS = (15,)  # validation/test scores kept for the operating-point section


# ----------------------------------------------------------------- shot model (as results/shots/shots.py)

def pauli_sets(n):
    """The readout sets, name -> PauliList. XYZ-strings: every Pauli string made of one letter (all orders)."""
    def strings(letter):
        return [Pauli("".join(letter if (m >> (n - 1 - q)) & 1 else "I" for q in range(n))) for m in range(1, 2 ** n)]

    return {"Z+ZZ": cli.build_observables("Z+ZZ", n),
            "XYZ-strings": PauliList(strings("X") + strings("Y") + strings("Z"))}


def settings_of(paulis):
    """Qubit-wise-commuting grouping -> list of (basis string, PauliList). Qubits no Pauli in the group touches are
    measured in Z. Basis strings are little-endian like Pauli labels (last character = qubit 0)."""
    out = []
    for group in paulis.group_qubit_wise_commuting():
        basis = ["Z"] * paulis.num_qubits
        for p in group:
            for q in range(paulis.num_qubits):
                if p.x[q] or p.z[q]:
                    basis[-(q + 1)] = str(p[q])
        out.append(("".join(basis), group))
    return out


def basis_change(basis):
    """Unitary that maps the measurement basis to the computational one (H for X, H·S† for Y)."""
    qc = QuantumCircuit(len(basis))
    for q, b in enumerate(reversed(basis)):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
    return Operator(qc).data


def eigenvalues(group, n):
    """±1 eigenvalue of every Pauli in the group on every outcome bitstring, (2^n, len(group))."""
    x = np.arange(2 ** n)
    masks = [sum(1 << q for q in range(n) if p.x[q] or p.z[q]) for p in group]
    return np.stack([1 - 2 * (np.bitwise_count(x & m).astype(int) & 1) for m in masks], 1).astype(float)


def shot_features(probs, settings, n, shots, rng):
    """Estimated expectation values, (M, n_paulis), in the order of the settings' groups. shots=None -> exact."""
    cols = []
    for basis, group in settings:
        p = probs[basis]
        if shots is not None:
            p = rng.multinomial(shots, p) / shots
        cols.append(p @ eigenvalues(group, n))
    return np.concatenate(cols, 1)


# ----------------------------------------------------------------- runs


class SettingsBackend(NumpyBackend):
    """NumpyBackend that also returns, per call, the features of every (set, budget) readout variant.

    Returns the exact expectation values of the CLI's observables as usual; `variants` collects one dict
    {(set, budget): (M, n_obs) float32} per run_batch call (train, validation, test in cli.qrc_features order)."""

    def __init__(self, seed):
        super().__init__(reset_inputs=True)
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
        chunk = max(1, self.max_chunk_bytes // (16 * D * D * 3))

        exact, out = [], {(s, b): [] for s in SETS for b in BUDGETS}
        for start in range(0, len(values), chunk):
            rho = self._density_matrices(values[start:start + chunk], gates, U, encoder.num_qubits)
            exact.append(np.einsum("pab,mba->mp", P, rho).real)
            probs = {b: np.einsum("xa,mab,xb->mx", V[b], rho, V[b].conj()).real for b in bases}
            probs = {b: np.clip(p, 0, None) / np.clip(p, 0, None).sum(1, keepdims=True) for b, p in probs.items()}
            for s in SETS:
                for budget in BUDGETS:
                    per_setting = None if budget is None else budget // len(settings[s])
                    out[(s, budget)].append(shot_features(probs, settings[s], n, per_setting, rng)
                                            .astype(np.float32))
        self.variants.append({k: np.concatenate(v) for k, v in out.items()})
        return np.concatenate(exact)


def qrc_args(reservoir, L, seed):
    argv = ["qrc", *RESERVOIR_ARGS[reservoir], *DATA, "--window-length", str(L), "--n-mem-qubits", "3",
            "--observables", OBSERVABLES, "--backend", "numpy", "--reset-inputs", "--seed", str(seed)]
    return cli.build_parser().parse_args(argv)


def evaluate(features, data, args):
    """cli.run's evaluation: readout + threshold on validation, Card Precision@K and per-category AP."""
    metrics, scores = cli.fit_and_evaluate(features, data, args.alpha, args.min_recall)
    cli.add_derived_metrics(metrics, scores, data, args.top_k)
    return metrics, scores


def run_qrc(reservoir, L, seed, save_scores):
    name = f"qrc_{reservoir}_L{L}_s{seed}"
    path = RUNS / f"{name}.json"
    if path.exists():
        print(f"{name}: exists, skipped")
        return
    args = qrc_args(reservoir, L, seed)
    data = cli.prepare_data(args)
    backend = SettingsBackend(seed)
    cli.make_backend = lambda *a, **k: backend
    t0 = time.perf_counter()
    exact, info = cli.qrc_features(data, args)
    seconds = time.perf_counter() - t0
    records = []
    for (s, budget) in backend.variants[0]:
        features = dict(zip(cli.SPLITS, (v[(s, budget)] for v in backend.variants)))
        if s == OBSERVABLES and budget is None:  # same observables as the CLI run, any order
            assert np.allclose(np.sort(features["test"][:200], 1), np.sort(exact["test"][:200], 1), atol=1e-5)
        t1 = time.perf_counter()
        metrics, scores = evaluate(features, data, args)
        records.append(dict(model="qrc", reservoir=reservoir, L=L, seed=seed, observables=s,
                            budget=budget, n_features=features["train"].shape[1], reservoir_seconds=seconds,
                            readout_seconds=time.perf_counter() - t1, metrics=metrics, **info))
        if save_scores:
            np.savez_compressed(RUNS / "scores" / f"{name}_{s.replace('+', '_')}_{budget or 'exact'}.npz",
                                **{f"scores_{k}": v for k, v in scores.items()},
                                **{f"y_{k}": data[f"y_{k}"] for k in scores},
                                **{f"category_{k}": data[f"category_{k}"] for k in scores})
        print(f"{name} {s} {budget or 'exact'}: customer AP val "
              f"{metrics['validation']['average_precision_customer']:.3f}", flush=True)
    path.write_text(json.dumps(records, indent=1))


def run_classical(model, L, seed, n_units=None):
    """baseline, lstm or esn (n_units neurons, default reservoir settings) through cli.run."""
    seeded = model in ("lstm", "esn")
    tag = f"esn{n_units}" if model == "esn" else model
    name = f"{tag}_L{L}" + (f"_s{seed}" if seeded else "")
    path = RUNS / f"{name}.json"
    if path.exists():
        print(f"{name}: exists, skipped")
        return
    argv = [model, *DATA, "--window-length", str(L), "--output-dir", str(HERE / "cli_runs" / f"{tag}_L{L}")]
    argv += ["--seed", str(seed)] if seeded else []
    argv += ["--n-units", str(n_units)] if model == "esn" else []
    args = cli.build_parser().parse_args(argv)
    data = cli.prepare_data(args)
    run_dir = cli.run(args, data)
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    path.write_text(json.dumps([dict(model=model, L=L, seed=seed if seeded else None,
                                     n_units=n_units,
                                     metrics=metrics, run_dir=str(run_dir))], indent=1))


def check():
    """Exact features from the stored distributions vs NumpyBackend, every reservoir; shot-noise feature error."""
    data = cli.prepare_data(qrc_args("random", 5, 42))
    small = {f"X_{s}": data["X_test"][:300] for s in cli.SPLITS}
    paulis = cli.build_observables(OBSERVABLES, 4)
    grouped = [p for _, g in settings_of(paulis) for p in g]
    order = [list(paulis).index(p) for p in grouped]
    for reservoir in RESERVOIR_ARGS:
        backend = SettingsBackend(42)
        cli.make_backend = lambda *a, **k: backend
        exact, _ = cli.qrc_features(small, qrc_args(reservoir, 5, 42))
        ours = backend.variants[0][(OBSERVABLES, None)]
        err = np.abs(ours - exact["test"][:, order]).max()
        assert err < 1e-5, f"{reservoir}: {err:.2e}"
        noisy = {s: np.abs(backend.variants[0][(s, 10_000)] - backend.variants[0][(s, None)]).mean() for s in SETS}
        print(f"{reservoir}: exact match ({err:.1e}); 10⁴-shot feature error "
              + ", ".join(f"{s} {e:.4f}" for s, e in noisy.items()))


# rough run time in seconds on a laptop core, from the L = 10 smoke runs, to balance the CI shards
COST = {"qrc": lambda L: 10 + 7 * L, "lstm": lambda L: 20 * L, "esn": lambda L: 2 + L, "baseline": lambda L: 13}


def jobs():
    """jobs.txt: the full grid, most expensive first (so a snake deal over the CI shards balances them)."""
    lines = []
    for L in WINDOW_LENGTHS:
        for reservoir in RESERVOIR_ARGS:
            lines += [f"qrc --reservoir {reservoir} --window-length {L} --seed {seed}"
                      + (" --save-scores" if L in SCORE_LENGTHS else "") for seed in SEEDS]
        lines += [f"lstm --window-length {L} --seed {seed}" for seed in SEEDS]
        lines += [f"esn --window-length {L} --seed {seed} --n-units {n}" for n in ESN_UNITS for seed in SEEDS]
        lines.append(f"baseline --window-length {L}")
    lines.sort(key=lambda line: -COST[line.split()[0]](int(line.split()[line.split().index("--window-length") + 1])))
    (HERE / "jobs.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs written to {HERE / 'jobs.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("qrc", "baseline", "lstm", "esn", "jobs", "check"))
    parser.add_argument("--reservoir", choices=tuple(RESERVOIR_ARGS))
    parser.add_argument("--window-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--n-units", type=int, default=ESN_UNITS[0], help="ESN neurons (esn only)")
    a = parser.parse_args()
    (RUNS / "scores").mkdir(parents=True, exist_ok=True)
    if a.command == "qrc":
        run_qrc(a.reservoir, a.window_length, a.seed, a.save_scores)
    elif a.command in ("baseline", "lstm", "esn"):
        run_classical(a.command, a.window_length, a.seed, a.n_units if a.command == "esn" else None)
    elif a.command == "jobs":
        jobs()
    else:
        check()
