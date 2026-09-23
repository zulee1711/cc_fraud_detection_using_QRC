from typing import List

from qiskit.quantum_info import Pauli

def build_observables(specs: str, num_qubits: int) -> List[Pauli]:
    """Build observables based on given specifications.
    For now only 'Z' (all single-qubit Z expectations, i.e. the <Z_i>)
    and 'Z+ZZ' (all single-qubit Z plus the two-qubit Z correlators on a linear chain, i.e. the <Z_i Z_{i+1}>)
    are implemented."""

    def _get_pauli_op(n, idx, label) -> Pauli:
        s = ['I'] * n
        s[-(idx + 1)] = label
        return Pauli("".join(s))

    def _get_pauli_zz(n, idx1, idx2) -> Pauli:
        s = ['I'] * n
        s[-(idx1 + 1)] = 'Z'
        if s[-(idx2 + 1)] == 'Z':
            s[-(idx2 + 1)] = 'I'
        else:
            s[-(idx2 + 1)] = 'Z'
        return Pauli("".join(s))

    observables = []

    if specs=="Z":
        for i in range(num_qubits):
            observables.append(_get_pauli_op(num_qubits, i, "Z"))
        return observables

    if specs=="Z+ZZ":
        for i in range(num_qubits):
            observables.append(_get_pauli_op(num_qubits, i, "Z"))
        for i in range(num_qubits-1):
            observables.append(_get_pauli_zz(num_qubits, i, i+1))
        return observables

    raise ValueError("Invalid observables specification")