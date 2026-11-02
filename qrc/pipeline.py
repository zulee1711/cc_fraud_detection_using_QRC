"""The experiment pipeline shared by the CLI, the tuning and sweep scripts, and the results scripts.

    data source → feature engineering → DataProcessor → customer windowing
        → features (raw window, random map, ESN or quantum reservoir)
        → logistic readout fitted on train, threshold on validation
        → validation and test metrics

Functions take an ``argparse.Namespace`` from ``experiments.cli.build_parser`` (or any object with the
same attributes). ``build_protocol`` and ``qrc_features`` accept an explicit ``backend`` and
``hamiltonian`` builder, so callers can swap either without touching module state.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from qrc.backends.estimator import EstimatorBackend
from qrc.backends.numpy_backend import NumpyBackend
from qrc.backends.spinpulse import SpinPulseBackend, SpinPulseConfig
from qrc.baselines.esn import EchoStateNetwork
from qrc.baselines.random_maps import SUPPORTED_RANDOM_MAPS, make_random_map
from qrc.datasets import load_split_files, split_dataset
from qrc.datasets.create_dataset import add_frauds, generate_dataset, scaled_simulation
from qrc.datasets.split_dataset import main as split_and_save_dataset
from qrc.decision import card_precision_top_k
from qrc.encodings import AngleEncoding, DenseAngleEncoding, ReuploadingEncoding
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

logger = get_logger(__name__)

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
    if simulator != "aer" and name not in ("gaussian", "shots"):
        raise ValueError(f"--simulator {simulator} only applies to --backend gaussian or shots")
    if name == "estimator-exact":
        return EstimatorBackend(reset_inputs=reset_inputs)
    if name == "numpy":
        return NumpyBackend(reset_inputs=reset_inputs)
    if name == "gaussian" and simulator == "numpy":
        # same values as Aer's (exact + the same Gaussian draws), much faster on few qubits
        return NumpyBackend(reset_inputs=reset_inputs, precision=precision, seed=seed)
    if name == "shots" and simulator == "numpy":
        # outcomes drawn from the exact distribution of each basis, much faster on few qubits
        return NumpyBackend(reset_inputs=reset_inputs, precision=precision, seed=seed, sampling="shots")
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


def build_protocol(args, backend=None, hamiltonian=None):
    """The encoder → reservoir → observables → backend stack the qrc arguments describe.

    Args:
        args: the qrc arguments.
        backend (Backend, optional): run on this backend instead of the one ``args.backend`` names.
        hamiltonian (callable, optional): ``(num_qubits, topology, rng) -> Hamiltonian``, used
            instead of ``HAMILTONIAN_BUILDERS[args.hamiltonian]`` (not with ``--hamiltonian random``,
            nor with ``--field`` other than 1).

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
    if args.hamiltonian == "random" and hamiltonian is not None:
        raise ValueError("a Hamiltonian builder does not apply to the random-circuit reservoir")
    field = getattr(args, "field", 1.0)  # absent from qrc arguments built before --field existed
    if field != 1.0 and (hamiltonian is not None or args.hamiltonian == "random"):
        raise ValueError("--field only applies to the built-in Hamiltonians")
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
        if hamiltonian is not None:
            hamiltonian = hamiltonian(
                n_input_qubits + args.n_mem_qubits, topology=topology, rng=args.seed
            )
        else:
            hamiltonian = HAMILTONIAN_BUILDERS[args.hamiltonian](
                n_input_qubits + args.n_mem_qubits,
                topology=topology,
                rng=args.seed,
                field=field,
            )
        reservoir = HamiltonianReservoir(
            hamiltonian,
            n_input_qubits,
            time=args.evolution_time,
            steps=args.trotter_steps,
        )
    observables = build_observables(args.observables, reservoir.num_qubits)
    if backend is None:
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
        "field": field if topology else "",
        "trotter_steps": args.trotter_steps if topology else "",
    }
    return protocol, reservoir_info


def qrc_features(data, args, backend=None, hamiltonian=None):
    """Run the reservoir once over each split (``backend`` and ``hamiltonian`` as in `build_protocol`).

    Returns:
        tuple: the reservoir outputs per split, and the resolved reservoir settings
        for the run record (see `build_protocol`).
    """
    protocol, reservoir_info = build_protocol(args, backend, hamiltonian)
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

