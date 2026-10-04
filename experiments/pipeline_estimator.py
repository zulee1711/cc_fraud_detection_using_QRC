"""End-to-end QRC pipeline, now with more backends

Backends (``--backends``):
    estimator-exact  EstimatorBackend, precision 0 (should match pipeline.py)
    gaussian         EstimatorBackend, exact values + Gaussian noise of width ``--precision``
    shots            EstimatorBackend, genuine shot sampling (~1/precision**2 shots)

Example:
    python -m experiments.pipeline_estimator --precision 0.05
"""
import argparse
import time

import pandas as pd

from qrc.backends.estimator import EstimatorBackend
from qrc.datasets import split_dataset
from qrc.datasets.create_dataset import add_frauds, generate_dataset
from qrc.encodings import AngleEncoding
from qrc.features import FeatureEngineer
from qrc.observables import build_observables
from qrc.processing import DataProcessor
from qrc.protocol import QRCProtocol
from qrc.readout import ClassicalReadout
from qrc.reservoirs import RandomCircuitReservoir
from qrc.sequences import window_by_customer_id

FEATURE_SET = [
    'TERMINAL_RISK_7D',
    'TERMINAL_RISK_CHANGE_7D_30D',
    'TERMINAL_RISK_CHANGE_7D_180D',
    'CUSTOMER_AMOUNT_RATIO_180D',
    'CUSTOMER_AMOUNT_RATIO_90D',
    'CUSTOMER_AMOUNT_DEVIATION_30D',
]
BACKEND_NAMES = ("estimator-exact", "gaussian", "shots")


def make_backend(name: str, precision: float, seed: int):
    if name == "estimator-exact":
        return EstimatorBackend()
    return EstimatorBackend(precision=precision, seed=seed, sampling=name)


def prepare_windows(window_length: int, data_seed: int) -> dict:
    SIMULATION = dict(
        n_customers=10,
        n_terminals=100,
        nb_days=365,
        start_date="2025-01-01",
        r=5,
        default_random_state=data_seed,
    )

    customer_profiles, terminal_profiles, transactions = generate_dataset(**SIMULATION)
    transactions = add_frauds(customer_profiles, terminal_profiles, transactions)
    train, validation, test = split_dataset(transactions, train_ratio=0.70, validation_ratio=0.15)

    full_data = pd.concat([train, validation, test], ignore_index=True)
    full_data.drop(columns=['TX_FRAUD_SCENARIO'], inplace=True)

    feature_engineer = FeatureEngineer(
        windows=(1, 7, 30, 90, 180),
        save=False,
    )

    train_features, validation_features, test_features = (
        feature_engineer.run(
            full_data,
            train,
            validation,
            test,
        )
    )

    manual_feature_sets = {
        6: FEATURE_SET,
    }

    processor = DataProcessor(
        feature_sets=manual_feature_sets,  # feature_sets,
    )

    input_data = processor.process(
        train_features,
        validation_features,
        test_features,
        feature_set=6,
    )

    return window_by_customer_id(input_data, window_length)


def run_backend(name, backend, args, splits):
    """Run one backend over the three splits, then fit and evaluate the readout."""
    encoder = AngleEncoding(num_qubits=args.n_input_qubits)
    reservoir = RandomCircuitReservoir(num_input_qubits=args.n_input_qubits,
                                       num_mem_qubits=args.n_mem_qubits,
                                       depth=args.depth, rng=args.seed)
    observables = build_observables(args.observables, reservoir.num_qubits)
    protocol = QRCProtocol(encoder, reservoir, observables, backend)

    start = time.perf_counter()
    features = {split: protocol.run(X) for split, (X, _) in splits.items()}
    elapsed = time.perf_counter() - start

    readout = ClassicalReadout(alpha=args.alpha)
    readout.fit(features["train"], splits["train"][1])
    metrics, *_ = readout.evaluate(features["test"], splits["test"][1])
    return {"backend": name, "features": features, "seconds": elapsed, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="+", choices=BACKEND_NAMES, default=list(BACKEND_NAMES))
    parser.add_argument("--precision", type=float, default=0.05,
                        help="target std. error of each expectation value (shots ~ 1/precision**2)")
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--n-input-qubits", type=int, default=6,
                        help="must equal the number of features (%d) unless you accept the tiling" % len(FEATURE_SET))
    parser.add_argument("--n-mem-qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--observables", default="Z")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = prepare_windows(args.window_length, data_seed=0)
    splits = {split: (data[f"X_{split}"], data[f"y_{split}"]) for split in ("train", "validation", "test")}
    for split, (X, y) in splits.items():
        print(f"{split:>10}: X {X.shape}, frauds {int(y.sum())}/{len(y)}")

    results = []
    for name in args.backends:
        backend = make_backend(name, args.precision, args.seed)
        print(f"running {name} ...", flush=True)
        results.append(run_backend(name, backend, args, splits))

    summary = pd.DataFrame([{
        "backend": r["backend"],
        "seconds": r["seconds"],
        "f1": r["f1_score"], "precision": r["precision"],
        "recall": r["recall"], "roc_auc": r["roc_auc"],
    } for r in results])
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
