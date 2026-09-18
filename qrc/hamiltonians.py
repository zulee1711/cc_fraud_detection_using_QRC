"""Reservoir Hamiltonians.

Every builder is meant to return a :class:`Hamiltonian`: an operator whose *form*
is fixed when the experiment is designed and whose *couplings* are drawn once,
from a seeded generator, when the reservoir is constructed.

The models follow the 'Design choices' page from the Notion.
"""


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


def tfim(num_qubits):
    """`1.` Textbook transverse-field Ising model.

        H = Σ J_i,i+1 X_i X_i+1  +  ν Σ Z_i
    """
    raise NotImplementedError


def tfim_longitudinal(num_qubits):
    """`2.` Ising model with an added longitudinal field, which breaks
    integrability and makes the dynamics genuinely chaotic.

        H = Σ J_i,i+1 X_i X_i+1  +  ν Σ Z_i  +  Σ g_i X_i
    """
    raise NotImplementedError


def tfim_native(num_qubits):
    """`3.` Model `2.` conjugated by Hadamards (X <-> Z), so the coupling term
    lands on SpinPulse's native RZZ. Same spectrum, cheaper circuit.

        H = Σ J_i,i+1 Z_i Z_i+1  +  ν Σ X_i  +  Σ g_i Z_i
    """
    raise NotImplementedError


def native_exchange(num_qubits):
    """`4.` ...
    """
    raise NotImplementedError
