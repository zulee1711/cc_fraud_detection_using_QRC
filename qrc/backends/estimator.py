import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2 as Estimator

from qrc.backends.base import Backend
from qiskit.quantum_info import PauliList
from qrc.encodings import Encoding
from qrc.reservoirs import Reservoir


class EstimatorBackend(Backend):
    """Uses the Estimator primitive from Qiskit Aer."""

    def __init__(self):
        self.estimator = Estimator()

    def _run_window(self, window: np.ndarray, encoder, reservoir, observables) -> np.ndarray:
        """Run the full circuit for the given window.
        Observables are measured only once, after the loop.
        Returns np array with shape ``(observables,)``."""
        if window.ndim != 2:
            raise ValueError("window must have shape (window_length, features)")
        circuit = QuantumCircuit(reservoir.num_qubits)
        for values in window:
            # 1. encode into the input qubits
            encoded_values = np.resize(np.asarray(values, dtype=float), reservoir.num_input_qubits)
            circuit.compose(encoder.encode(encoded_values),
                            qubits=range(reservoir.num_input_qubits), inplace=True)
            # 2. evolve the whole register
            circuit.compose(reservoir.circuit(), inplace=True)

            # 3. discard and re-prepare the input qubits
            # TODO
        estimator_pub = (circuit, observables)
        job = self.estimator.run([estimator_pub])
        result = job.result()
        return result[0].data.evs

    def run_batch(self, windows: np.ndarray, encoder: Encoding, reservoir: Reservoir, observables: PauliList) -> np.ndarray:
        return np.asarray([self._run_window(window, encoder, reservoir, observables) for window in windows])

