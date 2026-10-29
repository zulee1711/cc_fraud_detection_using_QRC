"""Multi-objective hyperparameter search over the QRC pipeline with Optuna.

    python -m experiments.tune --n-trials 100 --size medium --backend gaussian
    python -m experiments.tune --study-name tfim --hamiltonians tfim tfim-native --n-trials 30
    python -m experiments.tune --objectives qubits depth --seeds 1 2 3 --n-trials 50

Each trial picks a reservoir configuration (see ``suggest_params``), runs the
reservoir, fits the logistic readout on train and is scored on **validation**.
Next to that score, which is maximised, the search minimises the costs named by
``--objectives`` (see ``COSTS``). The result is a Pareto front: the trials no
other trial beats on every objective at once. The test split is never used to
pick a trial; its metrics are recorded per trial and reported for the front only.

Any option of ``experiments.cli qrc`` that the search does not set (data source,
feature set, window length, backend, precision, ...) is passed through and held
fixed for the whole study. The data is prepared once and shared by every trial.

The study is stored in ``{output_dir}/{study_name}/study.db`` and resumes when the
same command is run again. Each trial is also saved as an ordinary qrc run under
``{output_dir}/{study_name}/runs/`` (config, metrics, scores and summary.csv),
``trials.csv`` lists every trial with its parameters, metrics and costs, and
``pareto.json`` the Pareto front, each trial with the command that reproduces it.
"""

import argparse
import json
import math
import shlex
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from qiskit import transpile
from qiskit.transpiler import CouplingMap

import experiments.cli as cli
from qrc.backends.estimator import window_circuit
from qrc.hamiltonians import HamiltonianTopology

N_MEM_QUBITS = (2, 5)
# angle / re-uploading: input qubits beyond one per feature (extra qubits repeat features)
EXTRA_INPUT_QUBITS = 0
ENCODING_LAYERS = (2, 3)  # re-uploading only
# above pi, RY(scaling * u) folds back: two different values give the same state
ENCODING_SCALING = (np.pi / 4, np.pi)
DEPTH = (1, 6)  # random circuit only
EVOLUTION_TIME = (0.1, 10.0)  # Hamiltonians only, log scale
# Hamiltonians only; one step makes reset tfim / tfim-longitudinal blind to u -> 1 - u
TROTTER_STEPS = (2, 3)
ALPHA = (1e-1, 1e5)  # log scale
OBSERVABLES = ("Z", "Z+ZZ", "XYZ", "XYZ+XYZ-2body")
# shots objective only: ~10^3, 10^4, 10^5 shots. At 0.05 the reservoir memory is
# already lost in readout noise (results/armb/noreset_l5)
PRECISIONS = (0.03, 0.01, 0.003)

# Costs the search can minimise next to the validation metric (--objectives).
COSTS = {
    "qubits": "input + memory qubits",
    "depth": "depth of one window's circuit, transpiled to hardware gates (see circuit_cost)",
    "bases": "measurement bases: groups of qubit-wise commuting observables, "
    "each one a separate circuit execution",
    "shots": "shots per measurement basis, ~1/precision^2; searched over PRECISIONS "
    "(needs --backend gaussian, shots or spinpulse)",
    "wall_time": "seconds to run the reservoir over all splits (mean over seeds)",
}
NOISY_BACKENDS = ("gaussian", "shots", "spinpulse")

# Stand-in for a spin-qubit native gate set until the SpinPulse backend provides one.
TRANSPILE = dict(
    basis_gates=["rz", "sx", "x", "cz"], optimization_level=1, seed_transpiler=0
)

ENCODINGS = ("angle", "dense", "reupload")
HAMILTONIANS = ("random", *cli.HAMILTONIAN_BUILDERS)
TOPOLOGIES = tuple(topology.value for topology in HamiltonianTopology)

# The qrc options the search sets; passing one of them through is an error.
SEARCHED = (
    "encoding",
    "n_input_qubits",
    "n_mem_qubits",
    "encoding_layers",
    "encoding_scaling",
    "hamiltonian",
    "topology",
    "evolution_time",
    "trotter_steps",
    "depth",
    "entangler",
    "observables",
    "alpha",
)


def input_qubit_range(encoding, n_features, max_qubits):
    """Input qubits an encoding can use: dense angle encoding packs up to two features
    per qubit, the others need at least one qubit per feature."""
    if encoding == "dense":
        low, high = (n_features + 1) // 2, n_features - 1
    else:
        low, high = n_features, n_features + EXTRA_INPUT_QUBITS
    return low, min(high, max_qubits - N_MEM_QUBITS[0])


def feasible_encodings(encodings, n_features, max_qubits):
    feasible = []
    for encoding in encodings:
        low, high = input_qubit_range(encoding, n_features, max_qubits)
        if low <= high:
            feasible.append(encoding)
    return feasible


def suggest_params(
    trial, n_features, encodings, hamiltonians, topologies, max_qubits, precisions=None
):
    """One point of the search space, as ``experiments.cli qrc`` argument values.

    Parameters that only apply to one branch (the input qubit range of each
    encoding, re-uploading layers, random-circuit depth, Hamiltonian topology,
    evolution time and Trotter steps) are only suggested on that branch. The precision (shots) is
    only searched when `precisions` is given.
    """
    encoding = trial.suggest_categorical("encoding", encodings)
    n_input_qubits = trial.suggest_int(
        f"n_input_qubits_{encoding}",
        *input_qubit_range(encoding, n_features, max_qubits),
    )
    n_mem_qubits = trial.suggest_int(
        "n_mem_qubits", N_MEM_QUBITS[0], min(N_MEM_QUBITS[1], max_qubits - n_input_qubits)
    )
    params = dict(
        encoding=encoding,
        n_input_qubits=n_input_qubits,
        n_mem_qubits=n_mem_qubits,
        encoding_layers=(
            trial.suggest_int("encoding_layers", *ENCODING_LAYERS)
            if encoding == "reupload"
            else 1
        ),
        encoding_scaling=trial.suggest_float("encoding_scaling", *ENCODING_SCALING),
        hamiltonian=trial.suggest_categorical("hamiltonian", hamiltonians),
        observables=trial.suggest_categorical("observables", OBSERVABLES),
        alpha=trial.suggest_float("alpha", *ALPHA, log=True),
    )
    if params["hamiltonian"] == "random":
        params.update(
            depth=trial.suggest_int("depth", *DEPTH),
            entangler=trial.suggest_categorical("entangler", ("cx", "cry")),
        )
    else:
        params.update(
            topology=trial.suggest_categorical("topology", topologies),
            evolution_time=trial.suggest_float(
                "evolution_time", *EVOLUTION_TIME, log=True
            ),
            trotter_steps=trial.suggest_int("trotter_steps", *TROTTER_STEPS),
        )
    if precisions:
        params["precision"] = trial.suggest_categorical("precision", precisions)
    return params


def circuit_cost(protocol, window_length, n_features, reset_inputs):
    """Size of the circuit that runs for one window: `window_length` rounds of encoding
    and reservoir, transpiled to ``TRANSPILE``'s gates on a linear chain of qubits, so
    two-qubit gates between non-neighbouring qubits pay for their SWAPs.

    ``depth`` is the number of time steps when gates on disjoint qubits run in
    parallel: the longest chain of gates any qubit goes through.
    """
    circuit = window_circuit(
        protocol.encoder,
        protocol.reservoir,
        window_length,
        protocol.encoder.input_width(n_features),
        reset_inputs,
    )
    compiled = transpile(
        circuit, coupling_map=CouplingMap.from_line(circuit.num_qubits), **TRANSPILE
    )
    return {
        "depth": compiled.depth(),
        "two_qubit_gates": compiled.num_nonlocal_gates(),
        "two_qubit_depth": compiled.depth(lambda inst: inst.operation.num_qubits == 2),
    }


def objective(trial, base_args, data, search, metric, seeds, objectives):
    """Mean validation `metric` over the reservoir seeds, then each cost in `objectives`."""
    params = suggest_params(trial, **search)
    results, run_dirs, seconds = [], [], []
    for seed in seeds:
        args = argparse.Namespace(
            **{**vars(base_args), **params, "seed": seed, "optuna_trial": trial.number}
        )
        start = time.perf_counter()
        features, reservoir_info = cli.qrc_features(data, args)
        seconds.append(time.perf_counter() - start)
        metrics, scores = cli.fit_and_evaluate(
            features, data, args.alpha, args.min_recall
        )
        cli.add_derived_metrics(metrics, scores, data, args.top_k)
        extra = {"reservoir_seconds": seconds[-1]}
        run_dirs.append(
            cli.save_run(args, data, metrics, scores, extra, reservoir_info).name
        )
        results.append(metrics)

    # The costs do not depend on the data; take them from the first seed's circuit
    protocol, _ = cli.build_protocol(
        argparse.Namespace(**{**vars(base_args), **params, "seed": seeds[0]})
    )
    precision = params.get("precision", base_args.precision)
    costs = {
        "qubits": protocol.reservoir.num_qubits,
        **circuit_cost(
            protocol, base_args.window_length, search["n_features"], base_args.reset_inputs
        ),
        "observables": len(protocol.observables),
        "bases": len(protocol.observables.group_qubit_wise_commuting()),
        # 0: exact expectation values
        "shots": math.ceil(1 / precision**2) if base_args.backend in NOISY_BACKENDS else 0,
        "wall_time": float(np.mean(seconds)),
    }

    trial.set_user_attr("params", params)
    trial.set_user_attr("run_dirs", run_dirs)
    for name, value in costs.items():
        trial.set_user_attr(name, value)
    for split in cli.EVAL_SPLITS:
        for name in results[0][split]:
            trial.set_user_attr(
                f"{split}_{name}", float(np.mean([m[split][name] for m in results]))
            )
    return [trial.user_attrs[f"validation_{metric}"], *(costs[c] for c in objectives)]


def cli_flags(params):
    """``experiments.cli qrc`` flags that reproduce `params`."""
    flags = []
    for key, value in params.items():
        flags += [f"--{key.replace('_', '-')}", str(value)]
    return flags


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # --seed (passed through to qrc) must not be read as an abbreviation of --seeds
        allow_abbrev=False,
        epilog="Every other option is passed to `experiments.cli qrc`; "
        "see `python -m experiments.cli qrc --help`.",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--timeout", type=float, default=None, help="stop after this many seconds"
    )
    parser.add_argument("--study-name", default="qrc")
    parser.add_argument(
        "--output-dir", type=Path, default=cli.PROJECT_ROOT / "results" / "tune"
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite in the study directory)",
    )
    parser.add_argument(
        "--metric",
        choices=(*cli.METRIC_NAMES, *cli.CATEGORY_METRIC_NAMES),
        default="average_precision",
        help="validation metric to maximise; the per-category ones need data with "
        "fraud scenarios",
    )
    parser.add_argument(
        "--objectives",
        nargs="*",
        choices=tuple(COSTS),
        default=None,
        help="costs to minimise next to the metric; none for a single-objective search. "
        "Default: all, except shots with an exact backend. "
        + "; ".join(f"{name}: {meaning}" for name, meaning in COSTS.items()),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="reservoir seeds each trial is averaged over (default: qrc --seed)",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=12,
        help="upper bound on input + memory qubits (simulation cost grows as 2^n)",
    )
    parser.add_argument(
        "--encodings", nargs="+", choices=ENCODINGS, default=list(ENCODINGS)
    )
    parser.add_argument(
        "--hamiltonians", nargs="+", choices=HAMILTONIANS, default=list(HAMILTONIANS)
    )
    parser.add_argument(
        "--topologies", nargs="+", choices=TOPOLOGIES, default=list(TOPOLOGIES)
    )
    parser.add_argument(
        "--sampler-seed", type=int, default=0, help="seed of the TPE sampler"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="trials run in parallel threads, sharing the prepared data. Speeds up the "
        "numpy backend and gaussian with --simulator numpy (numpy releases the GIL); the Aer "
        "backends gain nothing",
    )
    return parser


def main(argv=None):
    args, qrc_argv = build_parser().parse_known_args(argv)
    qrc_parser = cli.build_parser()
    base_args = qrc_parser.parse_args(["qrc", *qrc_argv])
    noisy = base_args.backend in NOISY_BACKENDS
    if args.objectives is None:
        objectives = [c for c in COSTS if noisy or c != "shots"]
    else:
        objectives = list(dict.fromkeys(args.objectives))
    if "shots" in objectives and not noisy:
        raise SystemExit(
            "the shots objective needs a sampled backend: --backend gaussian, shots or spinpulse"
        )
    # Shots can only be minimised if the search chooses them
    searched = (*SEARCHED, "precision") if "shots" in objectives else SEARCHED

    defaults = vars(qrc_parser.parse_args(["qrc"]))
    fixed = [k for k in searched if getattr(base_args, k) != defaults[k]]
    if fixed:
        raise SystemExit(
            f"searched by the tuner, remove from the command line: {', '.join(fixed)}"
        )

    n_features = len(cli.FEATURE_SETS[base_args.feature_set])
    encodings = feasible_encodings(args.encodings, n_features, args.max_qubits)
    if not encodings:
        raise SystemExit(
            f"no encoding of {n_features} features fits in --max-qubits {args.max_qubits}"
        )
    search = dict(
        n_features=n_features,
        encodings=encodings,
        hamiltonians=args.hamiltonians,
        topologies=args.topologies,
        max_qubits=args.max_qubits,
        precisions=list(PRECISIONS) if "shots" in objectives else None,
    )
    seeds = args.seeds or [base_args.seed]

    study_dir = args.output_dir / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    base_args.output_dir = study_dir / "runs"
    storage = args.storage or f"sqlite:///{(study_dir / 'study.db').resolve().as_posix()}"

    metric_names = [f"validation_{args.metric}", *objectives]
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["maximize"] + ["minimize"] * len(objectives),
        load_if_exists=True,
    )
    with warnings.catch_warnings():
        # multivariate/group TPE models the conditional branches jointly; still marked experimental
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        # offset by the trials so far, so a resumed study does not redraw the same startup trials
        study.sampler = optuna.samplers.TPESampler(
            seed=args.sampler_seed + len(study.trials), multivariate=True, group=True
        )

    # A resumed study must keep scoring trials the same way, on the same data
    setup = {
        "qrc_args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(base_args).items()
            if k not in (*searched, "seed", "output_dir")
        },
        "metric": args.metric,
        "objectives": objectives,
        "seeds": seeds,
        "search": search,
    }
    if "setup" not in study.user_attrs:
        study.set_user_attr("setup", setup)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study.set_metric_names(metric_names)
    elif study.user_attrs["setup"] != setup:
        raise SystemExit(
            f"study {args.study_name!r} was created with a different setup; "
            "use a new --study-name, or the same options:\n"
            + json.dumps(study.user_attrs["setup"], indent=2)
        )

    data = cli.prepare_data(base_args)
    cli.print_data_summary(data)
    if args.metric in cli.CATEGORY_METRIC_NAMES:
        # The per-category AP is undefined (NaN) without frauds of that category
        category = args.metric.removeprefix("average_precision_")
        n_frauds = data["summary"]["validation"].get(f"frauds_{category}", 0)
        if not n_frauds:
            if not study.trials:
                # don't leave a study whose setup would block the corrected rerun
                optuna.delete_study(study_name=args.study_name, storage=storage)
            raise SystemExit(
                f"--metric {args.metric}: the validation split has no {category} frauds; "
                "use data with fraud scenarios and more customers (e.g. data/sc3_c1000_x6)"
            )

    study.optimize(
        lambda trial: objective(
            trial, base_args, data, search, args.metric, seeds, objectives
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
    )

    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not completed:
        print("No trial completed.")
        return study

    front = sorted(study.best_trials, key=lambda trial: -trial.values[0])
    pareto = []
    for trial in front:
        command = ["python", "-m", "experiments.cli", "qrc", *qrc_argv]
        command += cli_flags(trial.user_attrs["params"])
        if len(seeds) == 1:
            command += ["--seed", str(seeds[0])]
        pareto.append(
            {
                "trial": trial.number,
                "objectives": dict(zip(metric_names, trial.values)),
                **trial.user_attrs,
                "command": shlex.join(command),
            }
        )
    (study_dir / "pareto.json").write_text(json.dumps(pareto, indent=2))

    table = pd.DataFrame(
        [
            {
                "trial": entry["trial"],
                **entry["objectives"],
                "encoding": entry["params"]["encoding"],
                "reservoir": entry["params"]["hamiltonian"],
                "observables": entry["params"]["observables"],
                f"test_{args.metric}": entry[f"test_{args.metric}"],
            }
            for entry in pareto
        ]
    )
    print(f"\nPareto front: {len(front)} of {len(completed)} trials")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("Reproduce a trial with its command in pareto.json, e.g. the top one:")
    print("  " + pareto[0]["command"])
    print(f"Study saved to {study_dir}")
    return study

if __name__ == "__main__":
    main()
