"""End-to-end QRC pipeline."""

from qrc.encodings import AngleEncoding
from qrc.reservoirs import RandomCircuitReservoir
from qrc.backends.statevector import StatevectorBackend
from qrc.protocol import QRCProtocol
from qrc.readout import ClassicalReadout
from qrc.observables import build_observables


def run_random_reservoir_experiment(
        data,
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

    train_features = protocol.run(data.X_train)

    readout = ClassicalReadout(alpha=alpha)
    metrics, y_readout, predictions, _ = readout.fit_evaluate(
        train_features,
        data.y_train,
    )
    return {
        "readout": readout,
        "metrics": metrics,
    }