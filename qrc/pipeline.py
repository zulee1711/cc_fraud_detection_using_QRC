"""End-to-end QRC pipeline."""

from .encodings import AngleEncoding
from .reservoirs import RandomCircuitReservoir
from .backends.statevector import StatevectorBackend
from .protocol import QRCProtocol
from .readout import ClassicalReadout


def run_random_reservoir_experiment(data, n_input_qubits = 4, n_mem_qubits = 2, depth = 2, seed = 42, alpha = 1.0):
    encoder = AngleEncoding(num_qubits=n_input_qubits)
    reservoir = RandomCircuitReservoir(num_input_qubits=n_input_qubits, num_mem_qubits=n_mem_qubits, depth=depth, rng=seed)
    observables = [("Z", index) for index in range(n_input_qubits+n_mem_qubits)]
    backend = StatevectorBackend(observables)

    protocol = QRCProtocol(encoder, reservoir, backend)

    train_features = protocol.run(data.X_train)

    readout = ClassicalReadout(model_type="logistic", alpha=alpha)
    metrics, y_readout, predictions = readout.fit_evaluate(
        train_features,
        data.y_train,
    )
    return {
        "readout": readout,
        "metrics": metrics,
    }