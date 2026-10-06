"""Experiment runner for credit-card fraud detection with QRC.

Every command starts from the same data preparation:

    data source → feature engineering → DataProcessor → customer windowing

Data sources (``--data``):
    simulate  generate a dataset on the fly from a ``--size`` preset and ``--data-seed``.
              ``--save-data DIR`` also writes it in the SharePoint layout, so it can
              be shared and reloaded with ``--data load --data-dir DIR``.
    load      read ``DIR/transactions_{train,validation,test}.pkl``, e.g. the
              dataset downloaded from SharePoint into ``data/``.

Commands:
    prepare   data preparation only; prints split shapes and fraud counts.
    baseline  classical control: the same logistic readout, fitted on the raw
              (flattened) windows instead of the reservoir outputs.
    qrc       run the quantum reservoir, then fit the logistic readout.

The readout is fitted on train only. Validation and test metrics are both
reported; nothing is tuned on validation. Each baseline/qrc run is saved to
``{output_dir}/{timestamp}_{command}/`` and summarised in ``{output_dir}/summary.csv``.

Examples:
    python -m experiments.cli prepare --size small --save-data data/sim_small
    python -m experiments.cli baseline --size medium
    python -m experiments.cli qrc --size medium --backend estimator-exact
    python -m experiments.cli qrc --data load --data-dir data --backend gaussian --precision 0.05
"""
import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import qrc
from qrc.backends.estimator import EstimatorBackend
from qrc.backends.statevector import StatevectorBackend
from qrc.datasets import load_split_files, split_dataset
from qrc.datasets.create_dataset import add_frauds, generate_dataset
from qrc.datasets.split_dataset import main as split_and_save_dataset
from qrc.encodings import AngleEncoding
from qrc.features import FeatureEngineer
from qrc.logger import get_logger
from qrc.observables import build_observables
from qrc.processing import DataProcessor
from qrc.protocol import QRCProtocol
from qrc.readout import ClassicalReadout
from qrc.reservoirs import RandomCircuitReservoir
from qrc.sequences import SPLITS, window_by_customer_id

logger = get_logger(__name__)

PROJECT_ROOT = Path(qrc.__file__).resolve().parents[1]

# Feature sets from the feature analysis (on train only). Each set extends the previous one.
_SET_7 = [
    "TERMINAL_RISK_7D",
    "TERMINAL_RISK_CHANGE_7D_30D",
    "TERMINAL_RISK_CHANGE_7D_90D",
    "CUSTOMER_AMOUNT_RATIO_90D",
    "CUSTOMER_AMOUNT_DEVIATION_30D",
    "TX_AMOUNT_LOG",
    "TX_DURING_NIGHT",
]
_SET_10 = _SET_7 + [
    "TERMINAL_RISK_30D",
    "TERMINAL_RISK_CHANGE_1D_90D",
    "CUSTOMER_NB_TX_30D",
]
_SET_15 = _SET_10 + [
    "TERMINAL_RISK_1D",
    "CUSTOMER_AMOUNT_SHIFT_7D_30D",
    "CUSTOMER_ACTIVITY_AMOUNT_30D",
    "CUSTOMER_TIME_GAP_SECONDS",
    "TX_DURING_WEEKEND",
]
FEATURE_SETS = {7: _SET_7, 10: _SET_10, 15: _SET_15}
FEATURE_WINDOWS = (1, 7, 30, 90, 180)
TRAIN_RATIO, VALIDATION_RATIO = 0.70, 0.15

# ~2 transactions per customer per day, so medium is ~35k transactions over a year.
SIZES = {
    "small": dict(n_customers=10, n_terminals=100),
    "medium": dict(n_customers=50, n_terminals=500),
    "large": dict(n_customers=100, n_terminals=1000),
}
SIMULATION = dict(nb_days=365, start_date="2025-01-01", r=5)

BACKEND_NAMES = ("statevector", "estimator-exact", "gaussian", "shots")
METRIC_NAMES = ("pr_auc", "roc_auc", "f1_score", "precision", "recall")
EVAL_SPLITS = ("validation", "test")
SUMMARY_FIELDS = (
    # run and data
    "run", "commit", "command", "data", "size", "data_seed", "data_dir", "start_date", "end_date",
    "feature_set", "n_features", "window_length",
    *(f"{split}_{k}" for split in SPLITS for k in ("samples", "frauds")),
    # reservoir (qrc only)
    "backend", "backend_precision", "n_input_qubits", "n_mem_qubits", "n_qubits", "depth", "entangler",
    "observables", "n_observables", "seed", "reservoir_seconds",
    # readout and metrics
    "alpha", "threshold",
    *(f"{split}_{m}" for split in EVAL_SPLITS for m in METRIC_NAMES),
)


# --------------------------------------------------------------------------- data

def load_transactions(args):
    """Return the raw ``(train, validation, test)`` transactions from the chosen source."""
    if args.data == "load":
        data_dir = Path(args.data_dir)
        logger.info(f"Loading splits from {data_dir}")
        splits = load_split_files(data_dir, args.start_date, args.end_date)
        return tuple(splits[split] for split in SPLITS)

    simulation = dict(SIZES[args.size], **SIMULATION, default_random_state=args.data_seed)
    customer_profiles, terminal_profiles, transactions = generate_dataset(**simulation)
    transactions = add_frauds(customer_profiles, terminal_profiles, transactions)
    if args.save_data:
        # Same layout as the shared dataset, so it can be reloaded with --data load
        return split_and_save_dataset(transactions, TRAIN_RATIO, VALIDATION_RATIO, output_dir=args.save_data)
    return split_dataset(transactions, train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO)


def prepare_data(args):
    """The data preparation shared by every command.

    Returns:
        dict: the output of ``window_by_customer_id`` (``X_{split}`` of shape
        ``(n, window_length, features)``, ``y_{split}``, ...), plus ``summary``,
        the split sizes and fraud counts.
    """
    train, validation, test = load_transactions(args)

    full_data = pd.concat([train, validation, test], ignore_index=True)
    full_data = full_data.drop(columns=["TX_FRAUD_SCENARIO"], errors="ignore")  # avoid data leakage

    feature_engineer = FeatureEngineer(windows=FEATURE_WINDOWS, save=False)
    train_features, validation_features, test_features = feature_engineer.run(full_data, train, validation, test)

    processor = DataProcessor(feature_sets=FEATURE_SETS)
    input_data = processor.process(train_features, validation_features, test_features, feature_set=args.feature_set)

    data = window_by_customer_id(input_data, args.window_length)
    data["summary"] = {
        split: {"samples": len(data[f"y_{split}"]), "frauds": int(data[f"y_{split}"].sum())}
        for split in SPLITS
    }
    return data


def print_data_summary(data):
    for split in SPLITS:
        X, s = data[f"X_{split}"], data["summary"][split]
        print(f"{split:>10}: X {X.shape}, frauds {s['frauds']}/{s['samples']}")


# --------------------------------------------------------------------------- models

def make_backend(name, precision, seed):
    if name == "statevector":
        return StatevectorBackend()
    if name == "estimator-exact":
        return EstimatorBackend()
    return EstimatorBackend(precision=precision, seed=seed, sampling=name)


def baseline_features(data):
    """Raw windows, flattened ``(n, L, F) → (n, L·F)``: what the readout sees without a reservoir."""
    return {split: data[f"X_{split}"].reshape(len(data[f"X_{split}"]), -1) for split in SPLITS}


def qrc_features(data, args):
    """Run the reservoir once over each split.

    Returns:
        tuple: the reservoir outputs per split, and the resolved reservoir settings
        (qubit counts after defaults, number of observables) for the run record.
    """
    n_input_qubits = args.n_input_qubits or len(FEATURE_SETS[args.feature_set])
    encoder = AngleEncoding(num_qubits=n_input_qubits)
    reservoir = RandomCircuitReservoir(num_input_qubits=n_input_qubits, num_mem_qubits=args.n_mem_qubits,
                                       depth=args.depth, entangler=args.entangler, rng=args.seed)
    observables = build_observables(args.observables, reservoir.num_qubits)
    backend = make_backend(args.backend, args.precision, args.seed)
    protocol = QRCProtocol(encoder, reservoir, observables, backend)

    features = {}
    for split in SPLITS:
        print(f"running {args.backend} on {split} ...", flush=True)
        features[split] = protocol.run(data[f"X_{split}"])
    reservoir_info = {"n_input_qubits": n_input_qubits, "n_qubits": reservoir.num_qubits,
                      "n_observables": len(observables)}
    return features, reservoir_info


def fit_and_evaluate(features, data, alpha):
    """Fit the readout on train, evaluate on validation and test."""
    readout = ClassicalReadout(alpha=alpha)
    readout.fit(features["train"], data["y_train"])

    metrics, scores = {}, {}
    for split in ("validation", "test"):
        split_metrics, _, _, split_scores = readout.evaluate(features[split], data[f"y_{split}"])
        metrics[split] = {k: float(v) for k, v in split_metrics.items()}
        scores[split] = split_scores
    return metrics, scores


# --------------------------------------------------------------------------- results

def data_source(args):
    if args.data == "load":
        return {"data": "load", "data_dir": str(args.data_dir),
                "start_date": args.start_date, "end_date": args.end_date}
    return {"data": "simulate", "size": args.size, "data_seed": args.data_seed,
            **SIZES[args.size], **SIMULATION}


def git_commit():
    """Short hash of the checked-out commit, with ``-dirty`` if there are uncommitted changes; ``""`` outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return f"{commit}-dirty" if dirty else commit


def summary_row(args, data, metrics, reservoir_info, extra, run_name):
    """One summary.csv row: everything needed to identify, reproduce and compare the run."""
    row = {
        "run": run_name,
        "commit": git_commit(),
        "command": args.command,
        # start/end date only when loading: for a simulation, start_date is the simulator's, not a filter
        **{k: v for k, v in data_source(args).items()
           if k in (("data", "data_dir", "start_date", "end_date") if args.data == "load"
                    else ("data", "size", "data_seed"))},
        "feature_set": args.feature_set,
        "n_features": len(FEATURE_SETS[args.feature_set]),
        "window_length": args.window_length,
        **{f"{split}_{k}": data["summary"][split][k] for split in SPLITS for k in ("samples", "frauds")},
        "alpha": args.alpha,
        "threshold": metrics["test"]["threshold_used"],
        **{f"{split}_{m}": metrics[split][m] for split in EVAL_SPLITS for m in METRIC_NAMES},
        **extra,
    }
    if args.command == "qrc":
        row.update(
            backend=args.backend,
            # only the sampled backends use it
            backend_precision=args.precision if args.backend in ("gaussian", "shots") else "",
            n_mem_qubits=args.n_mem_qubits, depth=args.depth, entangler=args.entangler,
            observables=args.observables, seed=args.seed, **reservoir_info,
        )
    return row


def append_summary(summary_path, row):
    """Append `row` to summary.csv, writing the header if the file is new."""
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_run(args, data, metrics, scores, extra, reservoir_info=None):
    """Write config, metrics and scores to a fresh run directory and append to summary.csv."""
    output_dir = Path(args.output_dir)
    run_dir = output_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.command}"
    run_dir.mkdir(parents=True)

    row = summary_row(args, data, metrics, reservoir_info or {}, extra, run_dir.name)
    config = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "commit": row["commit"],
        "data_source": data_source(args),
        "features": FEATURE_SETS[args.feature_set],
        "reservoir": reservoir_info or {},
        "splits": data["summary"],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({**metrics, **extra}, indent=2))
    np.savez(run_dir / "scores.npz",
             **{f"y_{split}": data[f"y_{split}"] for split in scores},
             **{f"scores_{split}": s for split, s in scores.items()})

    append_summary(output_dir / "summary.csv", row)
    return run_dir


def print_metrics(metrics):
    table = pd.DataFrame({split: {m: metrics[split][m] for m in METRIC_NAMES} for split in ("validation", "test")}).T
    print()
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------------- CLI

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    data = common.add_argument_group("data")
    data.add_argument("--data", choices=("simulate", "load"), default="simulate",
                      help="simulate a dataset, or load the split .pkl files from --data-dir")
    data.add_argument("--size", choices=tuple(SIZES), default="medium", help="simulation preset")
    data.add_argument("--data-seed", type=int, default=0, help="simulation seed")
    data.add_argument("--save-data", type=Path, default=None,
                      help="also write the simulated splits here (reload with --data load --data-dir)")
    data.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                      help="directory holding transactions_{train,validation,test}.pkl")
    data.add_argument("--start-date", default=None, help="first day to load, YYYY-MM-DD (--data load)")
    data.add_argument("--end-date", default=None, help="last day to load, YYYY-MM-DD (--data load)")
    data.add_argument("--feature-set", type=int, choices=tuple(FEATURE_SETS), default=7,
                      help="selected features: 7, 10 or 15 (each extends the previous)")
    data.add_argument("--window-length", type=int, default=3, help="transactions per customer window")

    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--alpha", type=float, default=1.0, help="readout regularization (C = 1/alpha)")
    run.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "runs")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", parents=[common], help="data preparation only")
    commands.add_parser("baseline", parents=[common, run], help="logistic readout on the raw windows")

    qrc_parser = commands.add_parser("qrc", parents=[common, run], help="quantum reservoir + logistic readout")
    qrc_parser.add_argument("--backend", choices=BACKEND_NAMES, default="estimator-exact")
    qrc_parser.add_argument("--precision", type=float, default=0.05,
                            help="target std. error of each expectation value (gaussian/shots; shots ~ 1/precision**2)")
    qrc_parser.add_argument("--n-input-qubits", type=int, default=None,
                            help="defaults to the number of features in --feature-set")
    qrc_parser.add_argument("--n-mem-qubits", type=int, default=2)
    qrc_parser.add_argument("--depth", type=int, default=2)
    qrc_parser.add_argument("--entangler", choices=("cx", "cry"), default="cx")
    qrc_parser.add_argument("--observables", default="Z")
    qrc_parser.add_argument("--seed", type=int, default=42, help="reservoir and sampling seed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    data = prepare_data(args)
    print_data_summary(data)
    if args.command == "prepare":
        return None

    extra, reservoir_info = {}, None
    if args.command == "baseline":
        features = baseline_features(data)
    else:
        start = time.perf_counter()
        features, reservoir_info = qrc_features(data, args)
        extra["reservoir_seconds"] = time.perf_counter() - start

    metrics, scores = fit_and_evaluate(features, data, args.alpha)
    run_dir = save_run(args, data, metrics, scores, extra, reservoir_info)
    print_metrics(metrics)
    print(f"\nResults saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
