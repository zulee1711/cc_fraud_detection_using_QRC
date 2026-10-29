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
    baseline  logistic regression fitted on the raw (flattened) feature windows instead of the reservoir outputs.
    classical fixed random feature map (tanh or RFF) plus logistic readout.
    lstm      recurrent neural-network baseline fitted on the feature windows.
    esn       echo state network: a fixed random classical reservoir + the same logistic readout as qrc.
    qrc       run the quantum reservoir, then fit the logistic readout.

Models are fitted on train only. Validation loss selects the LSTM checkpoint;
validation and test metrics are reported. Each run is saved to
``{output_dir}/{timestamp}_{command}/`` and summarised in ``{output_dir}/summary.csv``.

Examples:
    python -m experiments.cli prepare --size small --save-data data/sim_small
    python -m experiments.cli baseline --size medium
    python -m experiments.cli classical --random-map-method rff --feature-set 20 --window-length 1
    python -m experiments.cli lstm --size medium
    python -m experiments.cli esn --size medium --n-units 50 --spectral-radius 0.9
    python -m experiments.cli qrc --size medium --backend estimator-exact
    python -m experiments.cli qrc --hamiltonian tfim --topology ring
    python -m experiments.cli qrc --data load --data-dir data --backend gaussian --precision 0.05
    python -m experiments.cli qrc --size small --backend spinpulse --noise pink --t2 50000 --precision 0

To run many configurations on the same prepared data, see ``experiments/sweep.py``.
"""

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import qrc
from qrc.backends.estimator import EstimatorBackend
from qrc.backends.numpy_backend import NumpyBackend
from qrc.backends.spinpulse import (
    DD_MODES,
    NOISE_TYPES,
    PULSE_SHAPES,
    SpinPulseBackend,
    SpinPulseConfig,
)
from qrc.datasets import load_split_files, split_dataset
from qrc.datasets.create_dataset import add_frauds, generate_dataset, scaled_simulation
from qrc.datasets.split_dataset import main as split_and_save_dataset
from qrc.decision import card_precision_top_k
from qrc.encodings import AngleEncoding, ReuploadingEncoding, DenseAngleEncoding
from qrc.evaluation import CATEGORIES, categorize, category_metrics
from qrc.features import FeatureEngineer
from qrc.hamiltonians import (
    HamiltonianTopology,
    native_exchange,
    tfim,
    tfim_longitudinal,
    tfim_native,
)
from qrc.logger import get_logger
from qrc.observables import build_observables
from qrc.processing import DataProcessor
from qrc.protocol import QRCProtocol
from qrc.readout import ClassicalReadout
from qrc.reservoirs import HamiltonianReservoir, RandomCircuitReservoir
from qrc.sequences import SPLITS, window_by_customer_id

from qrc.baselines.esn import EchoStateNetwork
from qrc.baselines.lstm import run_lstm_baseline
from experiments.classical_random_maps import (
    SUPPORTED_RANDOM_MAPS,
    make_random_map,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(qrc.__file__).resolve().parents[1]

# Raw per-transaction input, no customer or terminal history: any memory must come from the window
# (arm B, see docs/experiments.md).
_SET_1 = ["TX_AMOUNT_LOG"]
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
_SET_TX_AMOUNT_ENGINEERED = [
    "CUSTOMER_AMOUNT_DEVIATION_30D",
    "CUSTOMER_AMOUNT_RATIO_90D",
    "CUSTOMER_AMOUNT_SHIFT_7D_30D",
    "CUSTOMER_ACTIVITY_AMOUNT_30D",
    "TX_AMOUNT_LOG",
]
FEATURE_SETS = {
    1: _SET_1,
    7: _SET_7,
    10: _SET_10,
    15: _SET_15,
    20: _SET_TX_AMOUNT_ENGINEERED,
}
FEATURE_WINDOWS = (1, 7, 30, 90, 180)
# TRAIN_RATIO, VALIDATION_RATIO = 0.70, 0.15

# Number of customers, scaled down from the shared dataset (large, the same data with --data-seed 0)
# keeping its fraud rate (~0.9%) and scenario mix (qrc.datasets.create_dataset.scaled_simulation).
# ~700 transactions per customer per year: small ~14k transactions / ~130 frauds (smoke and
# end-to-end tests), medium ~170k / ~1.6k (the smallest with meaningful per-category metrics).
SIZES = {"small": 25, "medium": 250, "large": 5000}
SIMULATION = dict(nb_days=365, start_date="2025-01-01")

HAMILTONIAN_BUILDERS = {
    "tfim": tfim,
    "tfim-longitudinal": tfim_longitudinal,
    "tfim-native": tfim_native,
    "native-exchange": native_exchange,
}

BACKEND_NAMES = ("estimator-exact", "gaussian", "shots", "numpy", "spinpulse")
ENCODING_NAMES = ("auto", "angle", "dense", "reupload")
METRIC_NAMES = (
    "average_precision",
    "roc_auc",
    "f1_score",
    "precision",
    "recall",
    "false_positive_rate",
    "matthews_corr_coef",
    "balanced_accuracy",
    "card_precision_top_k",
)
# per fraud category (qrc.evaluation); only when the data carries TX_FRAUD_SCENARIO
CATEGORY_METRIC_NAMES = tuple(f"average_precision_{c}" for c in CATEGORIES)

EVAL_SPLITS = ("validation", "test")
SUMMARY_FIELDS = (
    # run and data
    "run",
    "commit",
    "command",
    "data",
    "size",
    "data_seed",
    "data_dir",
    "start_date",
    "end_date",
    "feature_set",
    "n_features",
    "window_length",
    *(
        f"{split}_{k}"
        for split in SPLITS
        for k in ("samples", "frauds", *(f"frauds_{c}" for c in CATEGORIES))
    ),
    # reservoir (qrc only)
    "backend",
    "backend_precision",
    "encoding",
    "encoding_layers",
    "encoding_scaling",
    "n_input_qubits",
    "n_mem_qubits",
    "n_qubits",
    "depth",
    "entangler",
    "reset_inputs",
    "observables",
    "n_observables",
    "seed",
    "reservoir_seconds",
    "hamiltonian",
    "topology",
    "evolution_time",
    "trotter_steps",
    # classical tanh random map
    "classical_model",
    "random_map_width",
    "random_map_scale",
    "random_map_seed",
    "rff_gamma",
    "classical_seconds",
    # lstm only
    "training_seconds",
    # esn only (also uses seed and reservoir_seconds)
    "n_units",
    "spectral_radius",
    "leak_rate",
    "input_scaling",
    "bias_scaling",
    "density",
    # readout and metrics
    "alpha",
    "threshold",
    "min_recall",
    "top_k",
    *(
        f"{split}_{m}"
        for split in EVAL_SPLITS
        for m in (*METRIC_NAMES, *CATEGORY_METRIC_NAMES)
    ),
)


# --------------------------------------------------------------------------- data


def load_transactions(args):
    """Return the raw ``(train, validation, test)`` transactions from the chosen source."""
    if args.data == "load":
        data_dir = Path(args.data_dir)
        logger.info(f"Loading splits from {data_dir}")
        splits = load_split_files(data_dir, args.start_date, args.end_date)
        return tuple(splits[split] for split in SPLITS)

    simulation, frauds = scaled_simulation(SIZES[args.size])
    customer_profiles, terminal_profiles, transactions = generate_dataset(
        **simulation, **SIMULATION, default_random_state=args.data_seed
    )
    transactions = add_frauds(customer_profiles, terminal_profiles, transactions, **frauds)
    if args.save_data:
        # Same layout as the shared dataset, so it can be reloaded with --data load
        return split_and_save_dataset(transactions, output_dir=args.save_data)
    return split_dataset(transactions)


def prepare_data(args):
    """The data preparation shared by every command.

    Returns:
        dict: the output of ``window_by_customer_id`` (``X_{split}`` of shape
        ``(n, window_length, features)``, ``y_{split}``, ...), plus ``summary``,
        the split sizes and fraud counts.
    """
    train, validation, test = load_transactions(args)

    full_data = pd.concat([train, validation, test], ignore_index=True)
    # kept aside for evaluation only, looked up by transaction since feature engineering reorders rows
    meta = (
        full_data.set_index("TRANSACTION_ID")[
            ["TX_FRAUD", "TX_FRAUD_SCENARIO", "TX_AMOUNT"]
        ]
        if "TX_FRAUD_SCENARIO" in full_data
        else None
    )
    full_data = full_data.drop(
        columns=["TX_FRAUD_SCENARIO"], errors="ignore"
    )  # avoid data leakage

    feature_engineer = FeatureEngineer(windows=FEATURE_WINDOWS, save=False)
    train_features, validation_features, test_features = feature_engineer.run(
        full_data, train, validation, test
    )

    processor = DataProcessor(feature_sets=FEATURE_SETS)
    input_data = processor.process(
        train_features, validation_features, test_features, feature_set=args.feature_set
    )

    data = window_by_customer_id(input_data, args.window_length)
    data["summary"] = {
        split: {
            "samples": len(data[f"y_{split}"]),
            "frauds": int(data[f"y_{split}"].sum()),
        }
        for split in SPLITS
    }
    if meta is not None:
        for split in SPLITS:
            rows = meta.loc[data[f"ids_{split}"]]
            category = categorize(
                rows["TX_FRAUD"], rows["TX_FRAUD_SCENARIO"], rows["TX_AMOUNT"]
            )
            if not np.array_equal(category != "legit", data[f"y_{split}"] == 1):
                raise ValueError(
                    f"{split}: fraud categories do not line up with the labels"
                )
            data[f"category_{split}"] = category
            for c in CATEGORIES:
                data["summary"][split][f"frauds_{c}"] = int((category == c).sum())
    return data


def print_data_summary(data):
    for split in SPLITS:
        X, s = data[f"X_{split}"], data["summary"][split]
        print(f"{split:>10}: X {X.shape}, frauds {s['frauds']}/{s['samples']}")


# --------------------------------------------------------------------------- models


def make_backend(name, precision, seed, reset_inputs=False, spinpulse=None, simulator="aer"):
    if simulator != "aer" and name != "gaussian":
        raise ValueError(f"--simulator {simulator} only applies to --backend gaussian")
    if name == "estimator-exact":
        return EstimatorBackend(reset_inputs=reset_inputs)
    if name == "numpy":
        return NumpyBackend(reset_inputs=reset_inputs)
    if name == "gaussian" and simulator == "numpy":
        # same values as Aer's (exact + the same Gaussian draws), much faster on few qubits
        return NumpyBackend(reset_inputs=reset_inputs, precision=precision, seed=seed)
    if name == "spinpulse":
        config = spinpulse or SpinPulseConfig(precision=precision, seed=seed)
        return SpinPulseBackend(config, reset_inputs=reset_inputs)
    return EstimatorBackend(
        precision=precision, seed=seed, sampling=name, reset_inputs=reset_inputs
    )


def resolve_encoding(name, n_input_qubits, n_features, encoding_layers):
    """The encoding ``auto`` stands for: re-uploading with several layers, else dense
    angle encoding when there are fewer input qubits than features, else angle encoding.
    """
    if name != "auto":
        return name
    if encoding_layers > 1:
        return "reupload"
    return "dense" if n_input_qubits < n_features else "angle"


def make_encoder(name, n_input_qubits, args):
    if name == "angle":
        return AngleEncoding(num_qubits=n_input_qubits, scaling=args.encoding_scaling)
    if name == "dense":
        return DenseAngleEncoding(
            num_qubits=n_input_qubits, scaling=args.encoding_scaling
        )
    return ReuploadingEncoding(
        num_qubits=n_input_qubits,
        num_layers=args.encoding_layers,
        scaling=args.encoding_scaling,
        rng=(args.seed, 1),
    )


def spinpulse_config(args):
    """
    SpinPulse hardware and noise settings from the qrc --hardware/--noise/--t2/... options.
    """
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
            tj_us=args.tj_us or None,  # 0 turns the coupling noise off
            **common,
        )
    return SpinPulseConfig(T2S=args.t2, TJS=args.tj, **common)

def baseline_features(data):
    """Raw windows, flattened ``(n, L, F) → (n, L·F)``: what the readout sees without a reservoir."""
    return {
        split: data[f"X_{split}"].reshape(len(data[f"X_{split}"]), -1)
        for split in SPLITS
    }


def classical_random_map_features(data, args):
    """Apply a fixed seeded classical random feature map to each split."""
    flattened = baseline_features(data)
    input_width = flattened["train"].shape[1]

    map_width = input_width if args.random_map_width is None else args.random_map_width

    method = args.random_map_method

    if method not in SUPPORTED_RANDOM_MAPS:
        raise ValueError(
            f"Unknown random map {method!r}. " f"Choose from {SUPPORTED_RANDOM_MAPS}."
        )

    random_map = make_random_map(
        method,
        input_width=input_width,
        map_width=map_width,
        scale=args.random_map_scale,
        seed=args.seed,
    )
    random_map.fit(flattened["train"])

    mapped = {split: random_map.transform(X) for split, X in flattened.items()}

    return mapped, random_map.get_metadata()


def build_protocol(args):
    """The encoder → reservoir → observables → backend stack the qrc arguments describe.

    Returns:
        tuple: the ``QRCProtocol``, and the resolved reservoir settings (qubit counts
        after defaults, number of observables) for the run record.
    """
    n_features = len(FEATURE_SETS[args.feature_set])
    n_input_qubits = args.n_input_qubits or n_features
    encoding = resolve_encoding(
        args.encoding, n_input_qubits, n_features, args.encoding_layers
    )
    encoder = make_encoder(encoding, n_input_qubits, args)
    encoder.input_width(n_features)
    if args.hamiltonian == "random":
        reservoir = RandomCircuitReservoir(
            num_input_qubits=n_input_qubits,
            num_mem_qubits=args.n_mem_qubits,
            depth=args.depth,
            entangler=args.entangler,
            rng=args.seed,
        )
        topology = ""
    else:
        topology = HamiltonianTopology(args.topology)
        hamiltonian = HAMILTONIAN_BUILDERS[args.hamiltonian](
            n_input_qubits + args.n_mem_qubits,
            topology=topology,
            rng=args.seed,
        )
        reservoir = HamiltonianReservoir(
            hamiltonian,
            n_input_qubits,
            time=args.evolution_time,
            steps=args.trotter_steps,
        )
    observables = build_observables(args.observables, reservoir.num_qubits)
    backend = make_backend(
        args.backend,
        args.precision,
        args.seed,
        args.reset_inputs,
        spinpulse_config(args) if args.backend == "spinpulse" else None,
        args.simulator,
    )
    protocol = QRCProtocol(encoder, reservoir, observables, backend)
    reservoir_info = {
        "encoding": encoding,
        "n_input_qubits": n_input_qubits,
        "n_qubits": reservoir.num_qubits,
        "n_observables": len(observables),
        "hamiltonian": args.hamiltonian,
        "topology": topology.value if topology else topology,
        "evolution_time": args.evolution_time if topology else "",
        "trotter_steps": args.trotter_steps if topology else "",
    }
    return protocol, reservoir_info


def qrc_features(data, args):
    """Run the reservoir once over each split.

    Returns:
        tuple: the reservoir outputs per split, and the resolved reservoir settings
        for the run record (see `build_protocol`).
    """
    protocol, reservoir_info = build_protocol(args)
    features = {}
    for split in SPLITS:
        print(f"running {args.backend} on {split} ...", flush=True)
        features[split] = protocol.run(data[f"X_{split}"])
        stderr = getattr(
            protocol.backend, "last_stderr", None
        )  # only SpinPulse sets it
        if stderr is not None:
            print(
                f"  noise-sampling std. error per feature: "
                f"mean {stderr.mean():.4f}, max {stderr.max():.4f}",
                flush=True,
            )
    return features, reservoir_info


def esn_features(data, args):
    """Run the echo state network once over each split (final reservoir state per window)."""
    esn = EchoStateNetwork(
        input_size=len(FEATURE_SETS[args.feature_set]),
        n_units=args.n_units,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak_rate,
        input_scaling=args.input_scaling,
        bias_scaling=args.bias_scaling,
        density=args.density,
        rng=args.seed,
    )
    return {split: esn.run(data[f"X_{split}"]) for split in SPLITS}


def fit_and_evaluate(features, data, alpha, min_recall):
    """Fit the readout on train, evaluate on validation and test."""
    readout = ClassicalReadout(alpha=alpha)
    readout.fit(features["train"], data["y_train"])
    readout.select_decision_threshold(
        features["validation"], data["y_validation"], min_recall=min_recall
    )

    metrics, scores = {}, {}
    for split in ("validation", "test"):
        split_metrics, _, _, split_scores = readout.evaluate(
            features[split], data[f"y_{split}"]
        )
        metrics[split] = {k: float(v) for k, v in split_metrics.items()}
        scores[split] = split_scores
    return metrics, scores


def add_operational_metrics(metrics, scores, data, top_k):
    """Add daily Card Precision@K to validation and test metrics."""
    for split in EVAL_SPLITS:
        metrics[split]["card_precision_top_k"] = card_precision_top_k(
            data[f"y_{split}"],
            scores[split],
            data[f"groups_{split}"],
            data[f"timestamps_{split}"],
            top_k=top_k,
        )


def add_derived_metrics(metrics, scores, data, top_k):
    """Add Card Precision@K and, when the data carries fraud categories, the per-category AP."""
    add_operational_metrics(metrics, scores, data, top_k)
    for split, split_scores in scores.items():
        if f"category_{split}" in data:
            metrics[split].update(
                category_metrics(
                    data[f"y_{split}"],
                    split_scores,
                    data[f"category_{split}"],
                )
            )


# --------------------------------------------------------------------------- results


def data_source(args):
    if args.data == "load":
        return {
            "data": "load",
            "data_dir": str(args.data_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
        }
    return {
        "data": "simulate",
        "size": args.size,
        "data_seed": args.data_seed,
        **{k: v for params in scaled_simulation(SIZES[args.size]) for k, v in params.items()},
        **SIMULATION,
    }


def git_commit():
    """Short hash of the checked-out commit, with ``-dirty`` if there are uncommitted changes; ``""`` outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
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
        **{
            k: v
            for k, v in data_source(args).items()
            if k
            in (
                ("data", "data_dir", "start_date", "end_date")
                if args.data == "load"
                else ("data", "size", "data_seed")
            )
        },
        "feature_set": args.feature_set,
        "n_features": len(FEATURE_SETS[args.feature_set]),
        "window_length": args.window_length,
        **{
            f"{split}_{k}": v
            for split in SPLITS
            for k, v in data["summary"][split].items()
        },
        "alpha": getattr(args, "alpha", ""),  # lstm has no logistic readout
        "threshold": metrics["validation"][
            "threshold_used"
        ],  # threshold is frozen and reused on test
        "min_recall": getattr(args, "min_recall", ""),
        "top_k": getattr(args, "top_k", ""),
        **{
            f"{split}_{m}": metrics[split][m]
            for split in EVAL_SPLITS
            for m in (*METRIC_NAMES, *CATEGORY_METRIC_NAMES)
            if m in metrics[split]
        },
        **extra,
    }
    if args.command == "qrc":
        row.update(
            backend=args.backend,
            # only the sampled backends use it
            backend_precision=(
                args.precision
                if args.backend in ("gaussian", "shots", "spinpulse")
                else ""
            ),
            encoding_layers=args.encoding_layers,
            encoding_scaling=args.encoding_scaling,
            n_mem_qubits=args.n_mem_qubits,
            depth=args.depth,
            entangler=args.entangler,
            reset_inputs=args.reset_inputs,
            observables=args.observables,
            seed=args.seed,
            **reservoir_info,
        )
    elif args.command == "lstm":
        row.update(seed=args.seed)
    elif args.command == "classical":
        row.update(seed=args.seed)
    elif args.command == "esn":
        row.update(
            n_units=args.n_units,
            spectral_radius=args.spectral_radius,
            leak_rate=args.leak_rate,
            input_scaling=args.input_scaling,
            bias_scaling=args.bias_scaling,
            density=args.density,
            seed=args.seed,
        )
    return row


def append_summary(summary_path, row):
    """Append `row` to summary.csv, writing the header if the file is new.

    A file written with an older set of columns is first rewritten under the
    current header, so new columns never shift the values of later rows.
    """
    write_header = not summary_path.exists()
    if not write_header:
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != SUMMARY_FIELDS:
                old_rows = list(reader)
                with open(summary_path, "w", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=SUMMARY_FIELDS, restval="")
                    writer.writeheader()
                    writer.writerows(old_rows)
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
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "commit": row["commit"],
        "data_source": data_source(args),
        "features": FEATURE_SETS[args.feature_set],
        "reservoir": reservoir_info or {},
        "splits": data["summary"],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({**metrics, **extra}, indent=2))
    np.savez(
        run_dir / "scores.npz",
        **{
            f"{key}_{split}": data[f"{key}_{split}"]
            for split in scores
            for key in ("y", "ids", "category")
            if f"{key}_{split}" in data
        },
        **{f"scores_{split}": s for split, s in scores.items()},
    )

    append_summary(output_dir / "summary.csv", row)
    return run_dir


def print_metrics(metrics):
    table = pd.DataFrame(
        {
            split: {m: metrics[split][m] for m in METRIC_NAMES}
            for split in ("validation", "test")
        }
    ).T
    print()
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    if CATEGORY_METRIC_NAMES[0] in metrics["test"]:
        by_category = pd.DataFrame(
            {
                split: {m: metrics[split][m] for m in CATEGORY_METRIC_NAMES}
                for split in ("validation", "test")
            }
        ).T
        print()
        print(by_category.to_string(float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------------- CLI


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    data = common.add_argument_group("data")
    data.add_argument(
        "--data",
        choices=("simulate", "load"),
        default="simulate",
        help="simulate a dataset, or load the split .pkl files from --data-dir",
    )
    data.add_argument(
        "--size", choices=tuple(SIZES), default="medium", help="simulation preset"
    )
    data.add_argument("--data-seed", type=int, default=0, help="simulation seed")
    data.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help="also write the simulated splits here (reload with --data load --data-dir)",
    )
    data.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="directory holding transactions_{train,validation,test}.pkl",
    )
    data.add_argument(
        "--start-date", default=None, help="first day to load, YYYY-MM-DD (--data load)"
    )
    data.add_argument(
        "--end-date", default=None, help="last day to load, YYYY-MM-DD (--data load)"
    )
    data.add_argument(
        "--feature-set",
        type=int,
        choices=tuple(FEATURE_SETS),
        default=7,
        help="selected features: 1 (raw log amount) or 7, 10, 15 (engineered, each extends the previous)",
    )
    data.add_argument(
        "--window-length", type=int, default=3, help="transactions per customer window"
    )

    run = argparse.ArgumentParser(add_help=False)
    run.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results" / "runs"
    )

    readout = argparse.ArgumentParser(add_help=False)
    readout.add_argument(
        "--alpha", type=float, default=1.0, help="readout regularization (C = 1/alpha)"
    )

    evaluation = argparse.ArgumentParser(add_help=False)
    evaluation.add_argument(
        "--min-recall",
        type=float,
        default=0.8,
        help=(
            "minimum validation fraud recall required when selecting the "
            "decision threshold that minimizes false positives"
        ),
    )
    evaluation.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="maximum customer/card alerts reviewed per day for Card Precision@K",
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", parents=[common], help="data preparation only")
    commands.add_parser(
        "baseline",
        parents=[common, run, readout, evaluation],
        help="logistic regression on the raw (flattened) feature windows",
    )
    classical_parser = commands.add_parser(
        "classical",
        parents=[common, run, readout, evaluation],
        help="logistic readout on a fixed classical random feature map",
    )
    classical_parser.add_argument(
        "--random-map-method",
        choices=SUPPORTED_RANDOM_MAPS,
        default="tanh",
        help="random feature map type",
    )
    classical_parser.add_argument(
        "--random-map-width",
        type=int,
        default=None,
        help="number of random-map features (defaults to flattened input width)",
    )
    classical_parser.add_argument(
        "--random-map-scale",
        type=float,
        default=1.0,
        help="scale of the random-map weights",
    )
    classical_parser.add_argument(
        "--seed", type=int, default=42, help="random-map seed"
    )

    lstm_parser = commands.add_parser(
        "lstm",
        parents=[common, run, evaluation],
        help="LSTM classifier on transaction windows",
    )
    lstm_parser.add_argument("--hidden-size", type=int, default=32)
    lstm_parser.add_argument("--epochs", type=int, default=50)
    lstm_parser.add_argument("--batch-size", type=int, default=256)
    lstm_parser.add_argument("--learning-rate", type=float, default=1e-3)
    lstm_parser.add_argument("--patience", type=int, default=8)
    lstm_parser.add_argument("--seed", type=int, default=42)

    esn_parser = commands.add_parser(
        "esn",
        parents=[common, run, readout, evaluation],
        help="echo state network (classical reservoir) + logistic readout",
    )
    esn_parser.add_argument("--n-units", type=int, default=50, help="reservoir neurons")
    esn_parser.add_argument(
        "--spectral-radius",
        type=float,
        default=0.9,
        help="largest |eigenvalue| of the recurrent weights",
    )
    esn_parser.add_argument(
        "--leak-rate",
        type=float,
        default=1.0,
        help="state update rate in (0, 1]; smaller = slower, longer memory",
    )
    esn_parser.add_argument(
        "--input-scaling", type=float, default=1.0, help="range of the input weights"
    )
    esn_parser.add_argument(
        "--bias-scaling", type=float, default=0.2, help="range of the neuron biases"
    )
    esn_parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="fraction of non-zero recurrent weights",
    )
    esn_parser.add_argument("--seed", type=int, default=42, help="reservoir seed")

    qrc_parser = commands.add_parser(
        "qrc",
        parents=[common, run, readout, evaluation],
        help="quantum reservoir + logistic readout",
    )
    qrc_parser.add_argument(
        "--backend",
        choices=BACKEND_NAMES,
        default="estimator-exact",
        help="'numpy' is exact like estimator-exact, much faster for small registers (up to ~8 qubits)",
    )
    qrc_parser.add_argument(
        "--precision",
        type=float,
        default=0.05,
        help="target std. error of each expectation value (gaussian/shots/spinpulse; "
        "shots ~ 1/precision**2; 0 = exact, spinpulse only)",
    )
    qrc_parser.add_argument(
        "--simulator",
        choices=("aer", "numpy"),
        default="aer",
        help="computes the exact values the gaussian backend adds noise to. 'numpy' gives "
        "the same features as 'aer' (same noise draws), much faster up to ~8 qubits",
    )
    qrc_parser.add_argument(
        "--n-input-qubits",
        type=int,
        default=None,
        help="defaults to the number of features in --feature-set",
    )
    qrc_parser.add_argument(
        "--encoding",
        choices=ENCODING_NAMES,
        default="auto",
        help="input encoding; 'auto' picks dense angle encoding with fewer input qubits "
        "than features, re-uploading with --encoding-layers > 1, else angle encoding",
    )
    qrc_parser.add_argument(
        "--encoding-layers",
        type=int,
        default=1,
        help="times each transaction is encoded (data re-uploading); 1 is plain angle encoding",
    )
    qrc_parser.add_argument(
        "--encoding-scaling",
        type=float,
        default=np.pi,
        help="angle per unit of (scaled) feature, in each encoding layer",
    )
    qrc_parser.add_argument("--n-mem-qubits", type=int, default=2)
    qrc_parser.add_argument(
        "--hamiltonian",
        choices=("random", *HAMILTONIAN_BUILDERS.copy().keys()),
        default="random",
        help="reservoir model; 'random' selects the random-circuit reservoir",
    )
    qrc_parser.add_argument(
        "--topology",
        choices=tuple(topology.value for topology in HamiltonianTopology),
        default=HamiltonianTopology.LINEAR.value,
        help="Hamiltonian connectivity (used unless --hamiltonian=random)",
    )
    qrc_parser.add_argument(
        "--evolution-time",
        type=float,
        default=1.0,
        help="Hamiltonian evolution time between encodings (unused for --hamiltonian=random)",
    )
    qrc_parser.add_argument(
        "--trotter-steps",
        type=int,
        default=2,
        help="Trotter steps per evolution (unused for --hamiltonian=random). One step "
        "makes reset tfim/tfim-longitudinal blind to u -> 1-u, so the default is 2",
    )
    qrc_parser.add_argument("--depth", type=int, default=2)
    qrc_parser.add_argument("--entangler", choices=("cx", "cry"), default="cx")
    qrc_parser.add_argument(
        "--reset-inputs",
        action="store_true",
        help="reset the input qubits before each encoding step (density-matrix simulation)",
    )
    qrc_parser.add_argument("--observables", default="Z")
    qrc_parser.add_argument(
        "--seed", type=int, default=42, help="reservoir and sampling seed"
    )

    spin = qrc_parser.add_argument_group(
        "spinpulse backend (only with --backend spinpulse)"
    )
    spin.add_argument(
        "--hardware",
        choices=("generic", "realistic"),
        default="generic",
        help="generic: the backend's default fields, times in arbitrary SpinPulse steps "
        "(--t2, --tj); realistic: the spin-qubit chip of SpinPulse's ParametrizingfromQPUSpecs "
        "example (1 us X/Y gates), times in us (--t2-us, --tj-us)",
    )
    spin.add_argument(
        "--noise",
        choices=NOISE_TYPES,
        default="pink",
        help="qubit-frequency noise: none, white, pink (1/f) or quasistatic",
    )
    spin.add_argument(
        "--t2",
        type=float,
        default=100.0,
        help="qubit dephasing time T2* in SpinPulse time units (shorter = stronger noise); "
        "compare it with the window's pulse duration, ~15,000 units for 9 qubits",
    )
    spin.add_argument(
        "--tj",
        type=float,
        default=None,
        help="dephasing time of the exchange coupling; off by default",
    )
    spin.add_argument(
        "--time-step-ns",
        type=float,
        default=10.0,
        help="realistic hardware: simulation time step in ns (5-10 is accurate)",
    )
    spin.add_argument(
        "--t2-us",
        type=float,
        default=10.0,
        help="realistic hardware: qubit dephasing time T2* in us (today's devices: 10)",
    )
    spin.add_argument(
        "--tj-us",
        type=float,
        default=5.0,
        help="realistic hardware: coupling dephasing time TJ* in us (today's devices: 5; 0 = off)",
    )

    spin.add_argument(
        "--only-idle",
        action="store_true",
        help="apply the frequency noise to idle qubits only",
    )
    spin.add_argument(
        "--dd",
        choices=DD_MODES,
        default="none",
        help="dynamical decoupling on idle qubits: none, spin-echo or full-drive",
    )
    spin.add_argument(
        "--pulse-shape",
        choices=PULSE_SHAPES,
        default="gaussian",
        help="envelope of the control pulses",
    )
    spin.add_argument(
        "--noise-samples",
        type=int,
        default=20,
        help="noise realizations averaged per window (accuracy vs. run time)",
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    data = prepare_data(args)
    print_data_summary(data)
    if args.command == "prepare":
        return None
    return run(args, data)


def run(args, data):
    """Fit and evaluate the model of `args.command` on the prepared `data`; returns the run directory."""
    extra, reservoir_info = {}, None
    if args.command == "baseline":
        features = baseline_features(data)
        metrics, scores = fit_and_evaluate(features, data, args.alpha, args.min_recall)
    elif args.command == "classical":
        start = time.perf_counter()
        features, classical_info = classical_random_map_features(data, args)
        extra.update(classical_info)
        metrics, scores = fit_and_evaluate(
            features,
            data,
            args.alpha,
            args.min_recall,
        )
        extra["classical_seconds"] = time.perf_counter() - start
    elif args.command == "lstm":
        splits = {split: (data[f"X_{split}"], data[f"y_{split}"]) for split in SPLITS}
        result = run_lstm_baseline(
            splits,
            seed=args.seed,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            min_recall=args.min_recall,
        )
        metrics, scores = result["metrics"], result["scores"]
        extra["training_seconds"] = result["seconds"]
    elif args.command == "esn":
        start = time.perf_counter()
        features = esn_features(data, args)
        extra["reservoir_seconds"] = time.perf_counter() - start
        metrics, scores = fit_and_evaluate(features, data, args.alpha, args.min_recall)
    else:
        start = time.perf_counter()
        features, reservoir_info = qrc_features(data, args)
        extra["reservoir_seconds"] = time.perf_counter() - start
        metrics, scores = fit_and_evaluate(
            features,
            data,
            args.alpha,
            args.min_recall,
        )

    add_derived_metrics(metrics, scores, data, args.top_k)
    run_dir = save_run(args, data, metrics, scores, extra, reservoir_info)
    print_metrics(metrics)
    print(f"\nResults saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
