"""Report section: TFIM (2 Trotter steps) vs an equal-width echo state network at equal tuning budget, L = 15.

Follows results/2_fraud (same data, encoding, readout, threshold and metrics, experiments.cli) and the protocol of
results/tuned_esn (Pandey, arXiv:2607.09905):
    - equal width: QRC read out with XYZ-strings on 4 qubits (45 observables), ESN with 45 units;
    - equal budget: one Optuna TPE study per model, same sampler seed, same number of trials (100), four
      hyperparameters each, each study starting from the model's untuned setting of results/2_fraud:
        TFIM  field h (multiplies the field terms of qrc.hamiltonians.tfim; h = 1 is the CLI TFIM), evolution
              time t, encoding angle scaling, readout alpha                    (2 Trotter steps, 1 + 3 qubits, reset)
        ESN   spectral radius, leak rate, input scaling, readout alpha         (bias 0.2, CLI density)
      objective = validation customer AP (exact expectation values for TFIM), mean over reservoir seeds 42-44;
      the test split is never read while tuning;
    - final numbers: the best setting on seeds 42-51 (45-51 held out), test split read once; TFIM evaluated exact
      and at 10⁵ / 10⁴ shots per window (results/2_fraud shot model). Validation/test scores are kept for the
      operational metrics.

Commands (from the repository root, PYTHONPATH=.):
    tuned.py tune --model {tfim,esn} [--trials N] [--workers W] [--max-minutes M]
                                    resumable (study_<model>.db); no new trial starts after M minutes
    tuned.py evaluate --model {tfim,esn} --seed S                   runs/<model>_tuned_s<S>.json, runs/scores/
    tuned.py time                                                    seconds per TFIM trial (3 seeds in parallel)
"""
import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from experiments import cli
from qrc.backends.numpy_backend import NumpyBackend
from qrc.evaluation import category_metrics
from qrc.hamiltonians import Hamiltonian, tfim
from qrc.readout import ClassicalReadout

_spec = importlib.util.spec_from_file_location("fraud", "results/2_fraud/fraud.py")
fraud = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fraud)

HERE = Path("results/3_tuned")
RUNS = HERE / "runs"
L = 15
READOUT = "XYZ-strings"
N_UNITS = 45  # = XYZ-strings observables on 4 qubits
TUNE_SEEDS = (42, 43, 44)
EVAL_SEEDS = tuple(range(42, 52))
SAMPLER_SEED = 0
TRIALS = 100
REFERENCE = {  # the untuned settings of results/2_fraud (CLI defaults)
    "tfim": dict(h=1.0, time=1.0, encoding_scaling=float(np.pi), alpha=1.0),
    "esn": dict(spectral_radius=0.9, leak_rate=1.0, input_scaling=1.0, alpha=1.0),
}


def suggest(trial, model):
    """The search spaces: four dimensions per model, log-uniform where the scale is unknown."""
    if model == "tfim":
        params = dict(h=trial.suggest_float("h", 0.01, 10.0, log=True),
                      time=trial.suggest_float("time", 0.1, 20.0, log=True),
                      encoding_scaling=trial.suggest_float("encoding_scaling", 0.1, 2 * np.pi, log=True))
    else:
        params = dict(spectral_radius=trial.suggest_float("spectral_radius", 0.05, 2.5),
                      leak_rate=trial.suggest_float("leak_rate", 0.01, 1.0, log=True),
                      input_scaling=trial.suggest_float("input_scaling", 0.01, 30.0, log=True))
    params["alpha"] = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
    return params


def use_field(h):
    """CLI TFIM with its field terms (the Z-only Paulis) multiplied by h; same couplings, same term order."""
    def build(num_qubits, topology, rng):
        H = tfim(num_qubits, topology=topology, rng=rng)
        coeffs = H.ops.coeffs.copy()
        coeffs[[not p.x.any() for p in H.ops.paulis]] *= h
        return Hamiltonian(f"tfim-h{h:.4g}", num_qubits, SparsePauliOp(H.ops.paulis, coeffs), {**H.params, "h": h})

    cli.HAMILTONIAN_BUILDERS["tfim"] = build


def cli_args(model, params, seed):
    """The CLI arguments of results/2_fraud at L = 15, with the tuned parameters."""
    if model == "tfim":
        use_field(params["h"])
        args = fraud.qrc_args("tfim", L, seed)
        args.evolution_time, args.encoding_scaling = params["time"], params["encoding_scaling"]
    else:
        args = cli.build_parser().parse_args(
            ["esn", *fraud.DATA, "--window-length", str(L), "--seed", str(seed), "--n-units", str(N_UNITS),
             "--spectral-radius", str(params["spectral_radius"]), "--leak-rate", str(params["leak_rate"]),
             "--input-scaling", str(params["input_scaling"])])
    args.alpha = params["alpha"]
    return args


_DATA = {}


def load_data():
    if not _DATA:
        _DATA.update(cli.prepare_data(cli_args("esn", REFERENCE["esn"], 42)))
    return _DATA


def tuning_features(model, params, seed):
    """Exact XYZ-strings expectation values (TFIM) or final ESN states, every split."""
    data, args = load_data(), cli_args(model, params, seed)
    if model == "esn":
        return cli.esn_features(data, args)
    args.observables = [str(p) for p in fraud.pauli_sets(4)[READOUT]]
    cli.make_backend = lambda *a, **k: NumpyBackend(reset_inputs=True)
    return cli.qrc_features(data, args)[0]


def validation_customer_ap(model, params, seed):
    """Readout fitted on train, scored on validation only."""
    data = load_data()
    feats = tuning_features(model, params, seed)
    readout = ClassicalReadout(alpha=params["alpha"])
    readout.fit(feats["train"], data["y_train"])
    scores = readout.predict(feats["validation"])
    return category_metrics(data["y_validation"], scores, data["category_validation"])["average_precision_customer"]


def study(model):
    """The model's Optuna study. The sampler seed is SAMPLER_SEED plus the trials already stored, so a resumed
    study does not replay the random draws of its first session (an uninterrupted study uses SAMPLER_SEED)."""
    import optuna

    storage = f"sqlite:///{HERE / f'study_{model}.db'}"
    try:
        n = len(optuna.load_study(study_name=model, storage=storage).trials)
    except KeyError:
        n = 0
    return optuna.create_study(study_name=model, storage=storage, direction="maximize", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED + n))


def tune(model, trials, workers, max_minutes=None):
    import optuna

    start = time.perf_counter()
    s = study(model)
    if not s.trials:
        s.enqueue_trial(REFERENCE[model])
    for t in s.trials:  # a trial cut off by a timeout is retried, not counted
        if t.state == optuna.trial.TrialState.RUNNING:
            s.enqueue_trial(t.params)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
    with ProcessPoolExecutor(workers, initializer=load_data) as pool:
        for _ in range(trials - done):
            if max_minutes and time.perf_counter() - start > 60 * max_minutes:
                print(f"{model}: {max_minutes} min reached, stopping (resume with the same command)")
                break
            trial = s.ask()
            params = suggest(trial, model)
            t0 = time.perf_counter()
            aps = list(pool.map(validation_customer_ap, *zip(*[(model, params, seed) for seed in TUNE_SEEDS])))
            trial.set_user_attr("seed_aps", aps)
            trial.set_user_attr("seconds", time.perf_counter() - t0)
            s.tell(trial, float(np.mean(aps)))
            print(f"{model} trial {trial.number}: {params} -> val customer AP {np.mean(aps):.4f} "
                  f"(best {s.best_value:.4f}, {time.perf_counter() - t0:.0f} s)", flush=True)


def evaluate(model, seed):
    path = RUNS / f"{model}_tuned_s{seed}.json"
    if path.exists():
        print(f"{path.name}: exists, skipped")
        return
    import optuna

    s = study(model)
    params = dict(s.best_params)
    n_trials = sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
    data, args = load_data(), cli_args(model, params, seed)
    t0 = time.perf_counter()
    if model == "tfim":  # exact + shot-sampled readouts from one run, as results/2_fraud
        backend = fraud.SettingsBackend(seed)
        cli.make_backend = lambda *a, **k: backend
        cli.qrc_features(data, args)
        variants = {b: dict(zip(cli.SPLITS, (v[(READOUT, b)] for v in backend.variants))) for b in fraud.BUDGETS}
    else:
        variants = {None: cli.esn_features(data, args)}
    seconds = time.perf_counter() - t0
    records = []
    for budget, feats in variants.items():
        metrics, scores = fraud.evaluate(feats, data, args)
        records.append(dict(model=model, L=L, seed=seed, params=params, n_trials=n_trials, observables=READOUT,
                            budget=budget, n_features=feats["train"].shape[1], reservoir_seconds=seconds,
                            metrics=metrics))
        np.savez_compressed(RUNS / "scores" / f"{model}_tuned_s{seed}_{budget or 'exact'}.npz",
                            **{f"scores_{k}": v for k, v in scores.items()})
    path.write_text(json.dumps(records, indent=1))
    print(f"{path.name}: test customer AP {records[0]['metrics']['test']['average_precision_customer']:.3f}")


def time_trial():
    """Seconds for one TFIM trial at the reference setting (3 seeds in parallel), and its validation AP."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(len(TUNE_SEEDS), initializer=load_data) as pool:
        aps = list(pool.map(validation_customer_ap, *zip(*[("tfim", REFERENCE["tfim"], s) for s in TUNE_SEEDS])))
    print(f"tfim reference: val customer AP {np.round(aps, 4)} mean {np.mean(aps):.4f}, "
          f"{time.perf_counter() - t0:.0f} s per trial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("tune", "evaluate", "time"))
    parser.add_argument("--model", choices=("tfim", "esn"))
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--workers", type=int, default=len(TUNE_SEEDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-minutes", type=float, help="tune: start no new trial after this")
    a = parser.parse_args()
    (RUNS / "scores").mkdir(parents=True, exist_ok=True)
    if a.command == "tune":
        tune(a.model, a.trials, a.workers, a.max_minutes)
    elif a.command == "evaluate":
        evaluate(a.model, a.seed)
    else:
        time_trial()
