"""Trotterised time evolution.

On gate-based hardware the free evolution exp(-iHt) of the reservoir has to be
approximated by a product formula.
"""


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
    raise NotImplementedError
