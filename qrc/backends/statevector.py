"""Exact statevector execution for batched QRC."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class StatevectorBackend:
    """Execute one independent temporal window at a time using exact states."""

    def _measure(self, state: Statevector, observables) -> np.ndarray:
        return np.asarray(
            [state.expectation_value(observable).real for observable in observables],
            dtype=float,
        )

    def run_window(self, window: np.ndarray, encoder, reservoir, observables) -> np.ndarray:
        """Run the full circuit for the given window.
        Observables are measured only once, after the loop.
        Returns np array with shape ``(observables,)``."""
        if window.ndim != 2:
            raise ValueError("window must have shape (window_length, features)")
        state = Statevector.from_label("0" * reservoir.num_qubits)
        for values in window:
            circuit = QuantumCircuit(reservoir.num_qubits)
            # 1. encode into the input qubits
            encoded_values = np.resize(np.asarray(values, dtype=float), reservoir.num_input_qubits)
            circuit.compose(encoder.encode(encoded_values),
                            qubits=range(reservoir.num_input_qubits), inplace=True)
            # 2. evolve the whole register
            circuit.compose(reservoir.circuit(), inplace=True)
            state = state.evolve(circuit)
            # 3. discard and re-prepare the input qubits
            # TODO
        measurements = self._measure(state, observables)
        return measurements

    def run_batch(self, windows: np.ndarray, encoder, reservoir, observables) -> np.ndarray:
        """Return reservoir trajectories with shape ``(samples, observables)``."""
        if windows.ndim != 3:
            raise ValueError("windows must have shape (samples, window_length, features)")
        return np.asarray([self.run_window(window, encoder, reservoir, observables) for window in windows])