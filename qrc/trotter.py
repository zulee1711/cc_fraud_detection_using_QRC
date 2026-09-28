"""Trotterised time evolution.

On gate-based hardware the free evolution exp(-iHt) of the reservoir has to be
approximated by a product formula.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from scipy.linalg import expm
import numpy as np


def pauli_rotation(qc, pauli, theta):
    """Adds a rotation exp(-i θ P) for a single Pauli term P to a circuit.

    Args:
        qc (QuantumCircuit): the circuit to append to.
        pauli (Pauli): the Pauli term to rotate under.
        theta (float): the rotation angle.
    """
    label = pauli.to_label()

    # ignore identity
    if set(label) == {"I"}:
        return

    active = [(i, p) for i, p in enumerate(reversed(label)) if p != "I"]

    # single-qubit
    if len(active) == 1:
        q, p = active[0]

        if p == "X":
            qc.rx(2 * theta, q)
        elif p == "Y":
            qc.ry(2 * theta, q)
        elif p == "Z":
            qc.rz(2 * theta, q)

    # two-qubit Pauli
    elif len(active) == 2:
        (q1, p1), (q2, p2) = active

        if p1 == "Z" and p2 == "Z":
            qc.rzz(2 * theta, q1, q2)

        elif p1 == "X" and p2 == "X":
            qc.h(q1)
            qc.h(q2)

            qc.rzz(2 * theta, q1, q2)

            qc.h(q1)
            qc.h(q2)

        elif p1 == "Y" and p2 == "Y":
            qc.rx(np.pi / 2, q1)
            qc.rx(np.pi / 2, q2)

            qc.rzz(2 * theta, q1, q2)

            qc.rx(-np.pi / 2, q1)
            qc.rx(-np.pi / 2, q2)

        else:
            raise ValueError(f"Unsupported Pauli term: {label}")

    else:
        raise ValueError(f"Only 1- and 2-qubit Pauli terms supported: {label}")


def trotter_circuit(hamiltonian, time, steps=1, decompose=True):
    """Builds the circuit approximating exp(-i H t) by first-order Lie-Trotter.

    Args:
        hamiltonian (Hamiltonian): the reservoir Hamiltonian to evolve under.
        time (float): evolution time between two encoding steps.
        steps (int): number of Trotter repetitions. At first order the error per
            step falls as (t/steps)^2, so this is the accuracy/depth knob.
        decompose (bool): if True, expand the evolution into rotation gates. Keep
            it True to count gates or hand the circuit to SpinPulse; False leaves
            a single opaque instruction, which draws more readably.

    Returns:
        QuantumCircuit: on `hamiltonian.num_qubits` qubits.
    """
    H = hamiltonian.ops
    dim = 2**H.num_qubits
    dt = time / steps

    if decompose:
        qc = QuantumCircuit(H.num_qubits)

        for _ in range(steps):
            for pauli, coeff in zip(H.paulis, H.coeffs):
                pauli_rotation(qc, pauli, coeff * dt)

        return qc

    else:
        U_step = np.eye(dim, dtype=complex)

        for pauli, coeff in zip(H.paulis, H.coeffs):
            P = pauli.to_matrix()
            U_j = expm(-1j * coeff * P * dt)
            U_step = U_j @ U_step

        U = np.linalg.matrix_power(U_step, steps)

        qc = QuantumCircuit(H.num_qubits)
        qc.unitary(U, range(H.num_qubits))

        return qc
