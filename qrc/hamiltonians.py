"""Reservoir Hamiltonians.

Every builder is meant to return a :class:`Hamiltonian`: an operator whose *form*
is fixed when the experiment is designed and whose *couplings* are drawn once,
from a seeded generator, when the reservoir is constructed.

The models follow the 'Design choices' page from the Notion.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


class Hamiltonian:
    """A fixed reservoir Hamiltonian on a linear chain of nearest neighbours.

    Attributes:
        name (str): short identifier, used when logging experiments.
        num_qubits (int): chain length (n_in + n_mem).
        ops (SparsePauliOp): the operator itself.
        params (dict): the coupling values that were drawn, so a reservoir can be
            reported and reproduced without re-deriving them from the seed.
    """

    def __init__(self, name, num_qubits, ops, params=None):
        self.name = name
        self.num_qubits = num_qubits
        self.ops = ops
        self.params = params or {}

    def __repr__(self):
        return f"Hamiltonian({self.name!r}, num_qubits={self.num_qubits}, terms={len(self.ops)})"


def tfim(num_qubits, v=1.0):
    """`1.` Textbook transverse-field Ising model.

    H = Σ J_i,i+1 X_i X_i+1  +  ν Σ Z_i
    """
    pauli_list = []
    params = {}

    for i in range(num_qubits - 1):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("XX", [i, i + 1], J))
        params[f"J{i}{i + 1}"] = J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-textbook", num_qubits, ops, params)


def tfim_longitudinal(num_qubits, v=1.0):
    """`2.` Ising model with an added longitudinal field, which breaks
    integrability and makes the dynamics genuinely chaotic.

        H = Σ J_i,i+1 X_i X_i+1  +  ν Σ Z_i  +  Σ g_i X_i
    """
    pauli_list = []
    params = {}

    for i in range(num_qubits - 1):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("XX", [i, i + 1], J))
        params[f"J{i}{i + 1}"] = J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))
        g = np.random.uniform(-1, 1)
        pauli_list.append(("X", [i], g))
        params[f"g{i}"] = g

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-longitudinal", num_qubits, ops, params)


def tfim_native(num_qubits, v=1.0):
    """`3.` Model `2.` conjugated by Hadamards (X <-> Z), so the coupling term
    lands on SpinPulse's native RZZ. Same spectrum, cheaper circuit.

        H = Σ J_i,i+1 Z_i Z_i+1  +  ν Σ X_i  +  Σ g_i Z_i
    """
    pauli_list = []
    params = {}

    for i in range(num_qubits - 1):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("ZZ", [i, i + 1], J))
        params[f"J{i}{i + 1}"] = J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))
        g = np.random.uniform(-1, 1)
        pauli_list.append(("X", [i], g))
        params[f"g{i}"] = g

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-native", num_qubits, ops, params)


def native_exchange(num_qubits, v=1.0, eta=1.0):
    """`4.` Extension of `3.` with a tunable XY term.

    H = Σ J_i,i+1 (Z_i Z_i+1 + η X_i X_i+1 + η Y_i Y_i+1)  +  ν Σ Z_i  +  Σ g_i X_i
    """
    pauli_list = []
    params = {}

    for i in range(num_qubits - 1):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("ZZ", [i, i + 1], J))
        params[f"J{i}{i + 1}"] = J
        pauli_list.append(("XX", [i, i + 1], eta * J))
        pauli_list.append(("YY", [i, i + 1], eta * J))
        params[f"etaJ{i}{i + 1}"] = eta * J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))
        g = np.random.uniform(-1, 1)
        pauli_list.append(("X", [i], g))
        params[f"g{i}"] = g

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-native", num_qubits, ops, params)
