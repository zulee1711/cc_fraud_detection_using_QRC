"""Exact statevector execution for batched QRC."""

import numpy as np
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
        """Return one trajectory with shape ``(window_length, observables)``."""
        if window.ndim != 2:
            raise ValueError("window must have shape (window_length, features)")
        state = Statevector.from_label("0" * reservoir.num_qubits)
        trajectory = []
        for values in window:
            encoded_values = np.resize(np.asarray(values, dtype=float), reservoir.num_qubits)
            circuit = encoder.encode(encoded_values).compose(reservoir.circuit())
            state = state.evolve(circuit)
            trajectory.append(self._measure(state))
        return np.asarray(trajectory)

    def run_batch(self, windows: np.ndarray, encoder, reservoir) -> np.ndarray:
        """Return reservoir trajectories with shape ``(samples, window, observables)``."""
        if windows.ndim != 3:
            raise ValueError("windows must have shape (samples, window_length, features)")
        return np.asarray([self.run_window(window, encoder, reservoir) for window in windows])