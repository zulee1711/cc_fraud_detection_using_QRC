"""Reservoir Hamiltonians.

Every builder is meant to return a :class:`Hamiltonian`: an operator whose *form*
is fixed when the experiment is designed and whose *couplings* are drawn once,
from a seeded generator, when the reservoir is constructed.

The models follow the 'Design choices' page from the Notion.
"""

from enum import Enum

import numpy as np
from qiskit.quantum_info import SparsePauliOp


class HamiltonianTopology(str, Enum):
    """Supported connectivity patterns for the reservoir Hamiltonians."""

    LINEAR = "linear"
    RING = "ring"
    FULLY_CONNECTED = "fully_connected"


def _interaction_pairs(num_qubits, topology):
    """Return the pairs of qubits that are coupled in the Hamiltonian.

    Args:
        num_qubits (int): the number of qubits in the reservoir.
        topology (HamiltonianTopology): the connectivity pattern.
    """
    if topology == HamiltonianTopology.LINEAR:
        return [(i, i + 1) for i in range(num_qubits - 1)]
    if topology == HamiltonianTopology.RING:
        if num_qubits <= 1:
            return []
        return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    if topology == HamiltonianTopology.FULLY_CONNECTED:
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    raise ValueError(f"Unsupported topology: {topology!r}")


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


def tfim(num_qubits, v=1.0, topology=HamiltonianTopology.LINEAR):
    """`1.` Textbook transverse-field Ising model.

    H = Σ J_i,j X_i X_j  +  ν Σ Z_i
    where the coupling pairs are selected according to `topology`.

    Args:
        num_qubits (int): the number of qubits in the reservoir.
        v (float): the strength of the transverse field, which sets the scale of the random couplings.
            Defaults to 1.0, which is what's used in the textbook TFIM literature.
        topology (HamiltonianTopology): the connectivity pattern. Default is linear which is the same
            as the Qoubly hardware topology.
    """
    pauli_list = []
    params = {}

    for i, j in _interaction_pairs(num_qubits, topology):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("XX", [i, j], J))
        params[f"J{i}{j}"] = J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-textbook", num_qubits, ops, params)


def tfim_longitudinal(num_qubits, v=1.0, topology=HamiltonianTopology.LINEAR):
    """`2.` Ising model with an added longitudinal field, which breaks
    integrability and makes the dynamics genuinely chaotic.

        H = Σ J_i,j X_i X_j  +  ν Σ Z_i  +  Σ g_i X_i

    Args:
        num_qubits (int): the number of qubits in the reservoir.
        v (float): the strength of the transverse field, which sets the scale of the random couplings.
        topology (HamiltonianTopology): the connectivity pattern.
    """
    pauli_list = []
    params = {}

    for i, j in _interaction_pairs(num_qubits, topology):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("XX", [i, j], J))
        params[f"J{i}{j}"] = J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))
        g = np.random.uniform(-1, 1)
        pauli_list.append(("X", [i], g))
        params[f"g{i}"] = g

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-longitudinal", num_qubits, ops, params)


def tfim_native(num_qubits, v=1.0, topology=HamiltonianTopology.LINEAR):
    """`3.` Model `2.` conjugated by Hadamards (X <-> Z), so the coupling term
    lands on SpinPulse's native RZZ. Same spectrum, cheaper circuit.

        H = Σ J_i,j Z_i Z_j  +  ν Σ X_i  +  Σ g_i Z_i

    Args:
        num_qubits (int): the number of qubits in the reservoir.
        v (float): the strength of the transverse field, which sets the scale of the random couplings.
        topology (HamiltonianTopology): the connectivity pattern.
    """
    pauli_list = []
    params = {}

    for i, j in _interaction_pairs(num_qubits, topology):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("ZZ", [i, j], J))
        params[f"J{i}{j}"] = J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))
        g = np.random.uniform(-1, 1)
        pauli_list.append(("X", [i], g))
        params[f"g{i}"] = g

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-native", num_qubits, ops, params)


def native_exchange(num_qubits, v=1.0, eta=1.0, topology=HamiltonianTopology.LINEAR):
    """`4.` Extension of `3.` with a tunable XY term.

    H = Σ J_i,j (Z_i Z_j + η X_i X_j + η Y_i Y_j)  +  ν Σ Z_i  +  Σ g_i X_i

    Args:
        num_qubits (int): the number of qubits in the reservoir.
        v (float): the strength of the transverse field, which sets the scale of the random couplings.
        eta (float): the relative strength of the XY term. Default is 1.
        topology (HamiltonianTopology): the connectivity pattern.
    """
    pauli_list = []
    params = {}

    for i, j in _interaction_pairs(num_qubits, topology):
        J = np.random.uniform(0, 1) * v
        pauli_list.append(("ZZ", [i, j], J))
        params[f"J{i}{j}"] = J
        pauli_list.append(("XX", [i, j], eta * J))
        pauli_list.append(("YY", [i, j], eta * J))
        params[f"etaJ{i}{j}"] = eta * J

    for i in range(num_qubits):
        pauli_list.append(("Z", [i], v))
        g = np.random.uniform(-1, 1)
        pauli_list.append(("X", [i], g))
        params[f"g{i}"] = g

    ops = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)

    return Hamiltonian("tfim-native", num_qubits, ops, params)
