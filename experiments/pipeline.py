"""End-to-end QRC pipeline."""

import pandas as pd

from qrc.encodings import AngleEncoding
from qrc.reservoirs import RandomCircuitReservoir
from qrc.backends.statevector import StatevectorBackend
from qrc.protocol import QRCProtocol
from qrc.readout import ClassicalReadout
from qrc.observables import build_observables

from qrc.datasets.create_dataset import generate_dataset, add_frauds
from qrc.features import FeatureEngineer
from qrc.processing import DataProcessor
from qrc.datasets import split_dataset
from qrc.sequences import window_by_customer_id


def run_random_reservoir_experiment(
        X_data,
        y_data,
        n_input_qubits = 4,
        n_mem_qubits = 2,
        depth = 2,
        observables_spec = "Z",
        seed = 42,
        alpha = 1.0
):
    encoder = AngleEncoding(num_qubits=n_input_qubits)
    reservoir = RandomCircuitReservoir(num_input_qubits=n_input_qubits, num_mem_qubits=n_mem_qubits, depth=depth, rng=seed)
    backend = StatevectorBackend()
    observables = build_observables(observables_spec, reservoir.num_qubits)

    protocol = QRCProtocol(encoder, reservoir, observables, backend)

    train_features = protocol.run(X_data)

    readout = ClassicalReadout(alpha=alpha)
    metrics, y_readout, predictions, _ = readout.fit_evaluate(
        train_features,
        y_data,
    )
    return {
        "readout": readout,
        "metrics": metrics,
    }

if __name__ == "__main__":
    SIMULATION = dict(
        n_customers=10,
        n_terminals=100,
        nb_days=365,
        start_date="2025-01-01",
        r=5,
        default_random_state=0,
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
        6: [
            'TERMINAL_RISK_7D',
            'TERMINAL_RISK_CHANGE_7D_30D',
            'TERMINAL_RISK_CHANGE_7D_180D',
            'CUSTOMER_AMOUNT_RATIO_180D',
            # 'TERMINAL_RISK_CHANGE_7D_90D',
            # 'TERMINAL_RISK_30D',
            'CUSTOMER_AMOUNT_RATIO_90D',
            'CUSTOMER_AMOUNT_DEVIATION_30D',
            # 'CUSTOMER_AMOUNT_MULTIPLIER_30D',
            # 'CUSTOMER_AMOUNT_RATIO_30D',
        ],
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

    L = 3
    windowed_input_data = window_by_customer_id(input_data, L)
    X_train = windowed_input_data['X_train']
    X_validation = windowed_input_data['X_validation']
    X_test = windowed_input_data['X_test']
    y_train = windowed_input_data['y_train']
    y_validation = windowed_input_data['y_validation']
    y_test = windowed_input_data['y_test']
    input_features = windowed_input_data['features']


    print(f"X_train {X_train.shape}, frauds {y_train.sum()}/{len(y_train)}")
    result = run_random_reservoir_experiment(X_train, y_train, n_input_qubits=6)
    print(result["metrics"])