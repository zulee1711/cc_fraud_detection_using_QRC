"""Exact statevector execution for batched QRC."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


class StatevectorBackend:
    """Execute one independent temporal window at a time using exact states."""

    def __init__(self, observables: list[tuple]):
        self.observables = observables

    def _observable(self, num_qubits: int, specification: tuple) -> Pauli:
        label = ["I"] * num_qubits
        kind = specification[0].upper()
        if kind in {"X", "Y", "Z"}:
            label[-(specification[1] + 1)] = kind
        elif kind == "ZZ":
            label[-(specification[1] + 1)] = "Z"
            label[-(specification[2] + 1)] = "Z"
        else:
            raise ValueError(f"Unsupported observable: {specification!r}")
        return Pauli("".join(label))

    def _measure(self, state: Statevector) -> np.ndarray:
        return np.asarray(
            [state.expectation_value(self._observable(state.num_qubits, item)).real for item in self.observables],
            dtype=float,
        )

    def run_window(self, window: np.ndarray, encoder, reservoir) -> np.ndarray:
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
        measurements = self._measure(state)
        return measurements

    def run_batch(self, windows: np.ndarray, encoder, reservoir) -> np.ndarray:
        """Return reservoir trajectories with shape ``(samples, observables)``."""
        if windows.ndim != 3:
            raise ValueError("windows must have shape (samples, window_length, features)")
        return np.asarray([self.run_window(window, encoder, reservoir) for window in windows])