import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import PauliList
from qiskit_aer.primitives import EstimatorV2 as Estimator

from qrc.backends.base import Backend
from qrc.encodings import Encoding
from qrc.reservoirs import Reservoir


class EstimatorBackend(Backend):
    """Uses the Estimator primitive from Qiskit Aer.

    The circuit is built once, with the window's features as a ``ParameterVector``,
    and all windows are submitted as the parameter values of a single PUB.
    """

    def __init__(self, precision=0.05):
        self.estimator = Estimator(options={
            "backend_options": {"method": "statevector"},   # later: "noise_model"
            "default_precision": precision,                       # shot noise; 0.0 = exact
        })

    def _build_circuit(self, encoder: Encoding, reservoir: Reservoir, window_length: int) -> QuantumCircuit:
        """Build the parametrized circuit for a window of ``window_length`` transactions.
        Has ``window_length * num_input_qubits`` parameters, ordered transaction by transaction."""
        n_in = reservoir.num_input_qubits
        theta = ParameterVector("x", window_length * n_in)
        reservoir_circuit = reservoir.circuit()
        circuit = QuantumCircuit(reservoir.num_qubits)
        for t in range(window_length):
            # 1. encode into the input qubits
            circuit.compose(encoder.encode(theta[t * n_in:(t + 1) * n_in]),
                            qubits=range(n_in), inplace=True)
            # 2. evolve the whole register
            circuit.compose(reservoir_circuit, inplace=True)
            # 3. discard and re-prepare the input qubits
            # TODO
        return circuit

    def run_batch(self, windows: np.ndarray, encoder: Encoding, reservoir: Reservoir, observables: PauliList) -> np.ndarray:
        """Run all windows with a single PUB. Observables are measured only once, after the last transaction.
        Returns np array with shape ``(samples, observables)``."""
        samples, window_length, num_features = windows.shape
        n_in = reservoir.num_input_qubits
        circuit = self._build_circuit(encoder, reservoir, window_length)

        # tile/truncate the features onto the input qubits
        resized = windows[..., np.arange(n_in) % num_features]

        parameter_values = resized.astype(float).reshape(samples, 1, window_length * n_in)

        job = self.estimator.run([(circuit, observables, parameter_values)])
        return np.asarray(job.result()[0].data.evs)
