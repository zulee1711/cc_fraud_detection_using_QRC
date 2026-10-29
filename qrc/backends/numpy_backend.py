"""Exact batched simulation in plain NumPy, for small registers."""
from typing import Optional

import numpy as np
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import Operator, PauliList

from qrc.backends.base import Backend
from qrc.encodings import Encoding
from qrc.reservoirs import Reservoir

# exp(-i θ/2 σ) for the encoding rotations
_PAULI = {
    "rx": np.array([[0, 1], [1, 0]], complex),
    "ry": np.array([[0, -1j], [1j, 0]], complex),
    "rz": np.array([[1, 0], [0, -1]], complex),
}


class NumpyBackend(Backend):
    """Exact expectation values, every window of a chunk simulated at once with dense matrices.

    Same protocol and qubit order as ``EstimatorBackend`` with precision 0: per transaction,
    optionally reset the input qubits, encode, then apply the reservoir unitary. Without reset the
    state is a statevector, with reset a density matrix.

    The reservoir unitary is computed once. Per window and step, the work is one 2^n × 2^n matrix
    product (two with reset), so the cost grows as 4^n (8^n with reset) per window. This is much
    faster than Aer for the 4-6 qubit registers used here and stops paying off around 9 qubits.

    Only encodings made of single-qubit rx/ry/rz gates are supported, each angle a linear function
    of at most one input value (all encodings in ``qrc.encodings``). No circuit noise.

    Args:
        reset_inputs (bool): reset the input qubits to |0> before each encoding step after the
            first, as in ``EstimatorBackend``.
        max_chunk_bytes (int): memory budget for the states of one chunk of windows; windows are
            processed in chunks that fit in it.
        precision (float): standard deviation of the Gaussian noise added to every expectation
            value, ``0.0`` for exact values. The noise is drawn exactly as in
            ``EstimatorBackend(sampling="gaussian")``, so with the same seed both give the same
            features up to machine precision.
        seed (int, optional): seed of the noise, for reproducible results.
    """

    def __init__(
        self,
        reset_inputs: bool = False,
        max_chunk_bytes: int = 256 * 2**20,
        precision: float = 0.0,
        seed: Optional[int] = None,
    ):
        if max_chunk_bytes < 1:
            raise ValueError(f"max_chunk_bytes must be positive, got {max_chunk_bytes}")
        if precision < 0:
            raise ValueError("precision must be non-negative")
        self.reset_inputs = reset_inputs
        self.max_chunk_bytes = max_chunk_bytes
        self.precision = precision
        # one stream for the whole lifetime of the backend: every call draws new noise
        self._rng = np.random.default_rng(seed)

    def run_batch(
        self,
        windows: np.ndarray,
        encoder: Encoding,
        reservoir: Reservoir,
        observables: PauliList,
    ) -> np.ndarray:
        """Returns np array with shape ``(samples, observables)``."""
        samples, window_length, num_features = windows.shape
        if samples == 0:
            return np.empty((0, len(observables)))
        input_width = encoder.input_width(num_features)
        gates = _encoding_gates(encoder, input_width)
        # same cyclic repetition of the features as EstimatorBackend
        values = windows[..., np.arange(input_width) % num_features].astype(float)

        U = Operator(reservoir.circuit()).data
        P = observables.to_matrix(array=True)
        D = U.shape[0]
        # a density matrix plus the step matrices, complex128
        per_window = 16 * D * D * (3 if self.reset_inputs else 2)
        chunk = max(1, self.max_chunk_bytes // per_window)

        out = []
        for start in range(0, samples, chunk):
            x = values[start : start + chunk]
            if self.reset_inputs:
                rho = self._density_matrices(x, gates, U, encoder.num_qubits)
                out.append(np.einsum("pab,mba->mp", P, rho).real)
            else:
                psi = self._statevectors(x, gates, U, encoder.num_qubits)
                out.append(np.einsum("ma,pab,mb->mp", psi.conj(), P, psi).real)
        evs = np.concatenate(out)
        if self.precision > 0:
            evs = evs + self._rng.normal(0.0, self.precision, evs.shape)
        return evs

    @staticmethod
    def _step_unitaries(x_t, gates, U, num_input_qubits):
        """U · (I_mem ⊗ V_in) for every window, shape (M, D, D)."""
        V = _input_unitaries(x_t, gates, num_input_qubits)
        D, d_in = U.shape[0], V.shape[1]
        M = len(V)
        full = np.einsum("ab,mij->maibj", np.eye(D // d_in), V).reshape(M, D, D)
        return U @ full

    def _statevectors(self, x, gates, U, num_input_qubits):
        M, L, _ = x.shape
        psi = np.zeros((M, U.shape[0]), complex)
        psi[:, 0] = 1
        for t in range(L):
            W = self._step_unitaries(x[:, t], gates, U, num_input_qubits)
            psi = np.einsum("mab,mb->ma", W, psi)
        return psi

    def _density_matrices(self, x, gates, U, num_input_qubits):
        M, L, _ = x.shape
        D, d_in = U.shape[0], 2**num_input_qubits
        rho = np.zeros((M, D, D), complex)
        rho[:, 0, 0] = 1
        for t in range(L):
            if t > 0:
                # trace out the input qubits (the low bits) and re-prepare them in |0...0>
                r = rho.reshape(M, D // d_in, d_in, D // d_in, d_in)
                rho = np.zeros_like(rho)
                rho.reshape(M, D // d_in, d_in, D // d_in, d_in)[:, :, 0, :, 0] = np.einsum("maibi->mab", r)
            W = self._step_unitaries(x[:, t], gates, U, num_input_qubits)
            rho = W @ rho @ W.conj().transpose(0, 2, 1)
        return rho


def _encoding_gates(encoder, input_width):
    """The encoding as a list of ``(qubit, pauli, slope, index, offset)``: the gate is
    exp(-i (slope·x[index] + offset)/2 · pauli), with ``index`` None for a fixed angle."""
    theta = ParameterVector("x", input_width)
    circuit = encoder.encode(theta)
    gates = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in _PAULI or len(instruction.qubits) != 1:
            raise NotImplementedError(f"NumpyBackend supports single-qubit rx/ry/rz encodings, got {name!r}")
        qubit = circuit.find_bit(instruction.qubits[0]).index
        angle = instruction.operation.params[0]
        if not isinstance(angle, ParameterExpression) or not angle.parameters:
            gates.append((qubit, _PAULI[name], 0.0, None, float(angle)))
            continue
        if len(angle.parameters) != 1:
            raise NotImplementedError(f"each encoding angle must depend on one input value, got {angle}")
        (parameter,) = angle.parameters
        offset = float(angle.bind({parameter: 0.0}))
        slope = float(angle.bind({parameter: 1.0})) - offset
        if not np.isclose(float(angle.bind({parameter: 2.0})), offset + 2 * slope):
            raise NotImplementedError(f"encoding angles must be linear in the input, got {angle}")
        gates.append((qubit, _PAULI[name], slope, parameter.index, offset))
    return gates


def _input_unitaries(x_t, gates, num_input_qubits):
    """The encoding unitary of one step for every window, (M, 2^n_in, 2^n_in), qubit 0 the lowest bit."""
    M = len(x_t)
    per_qubit = [np.broadcast_to(np.eye(2, dtype=complex), (M, 2, 2)) for _ in range(num_input_qubits)]
    for qubit, pauli, slope, index, offset in gates:
        angle = offset + (slope * x_t[:, index] if index is not None else np.zeros(M))
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        rotation = c[:, None, None] * np.eye(2) - 1j * s[:, None, None] * pauli
        per_qubit[qubit] = rotation @ per_qubit[qubit]
    V = per_qubit[0]
    for gate in per_qubit[1:]:
        # qubit q is more significant than all lower ones: V ← gate ⊗ V
        V = np.einsum("mab,mij->maibj", gate, V).reshape(M, 2 * V.shape[1], 2 * V.shape[1])
    return V
