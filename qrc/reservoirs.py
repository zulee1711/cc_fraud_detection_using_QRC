"""Quantum reservoirs.

A reservoir is understood to be the fixed unitary that acts on the whole register (input qubits +
memory qubits) between two encoding steps. Its structure is chosen when the
experiment is designed and its parameters are drawn once, from a seed, when it is
constructed — after that it never changes, which is the whole point of QRC: only the classical readout is trained.

Two kinds will be implemented, covering the five reservoirs described in the 'Design choices' page from the Notion:

* :class:`RandomCircuitReservoir` — `0.`, a fixed random circuit, no Hamiltonian
  involved. This is directly taken from the QRC-Lab and will serve as part of our MVP.
* :class:`HamiltonianReservoir` — `1.` to `4.`, the Trotterised evolution under a
  Hamiltonian from ``hamiltonians.py``. Those four models differ only in which
  Pauli terms they carry, so they are one class parameterised by an operator.

Every reservoir exposes the same ``circuit()``, an instance of Qiskit's QuantumCircuit.
"""

import numpy as np
from qiskit import QuantumCircuit

from . import trotter


class Reservoir:
    """Base class: a fixed unitary on `num_qubits` qubits.

    The circuit is identical at every layer of the protocol, so it is built once
    on first use and cached — over a long time series this saves rebuilding the
    same object T times.
    """

    name = "reservoir"

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self._circuit = None

    def _build(self):
        raise NotImplementedError

    def circuit(self):
        """Returns the reservoir's QuantumCircuit (cached)."""
        if self._circuit is None:
            self._circuit = self._build()
        return self._circuit

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, num_qubits={self.num_qubits})"


class RandomCircuitReservoir(Reservoir):
    """`0.` Fixed random circuit — entangling layer, then random local rotations.

    Entangles on a ring: from every qubit to the next, wrapping round.
    `entangler="cx"` reproduces the ``RandomReservoir`` from the QRC-Lab toolbox,
    while `entangler="cry"` is ``RandomCRotReservoir``.

    Note: this reservoir has no notion of evolution time — it is a circuit, not a
    dynamics, so depth plays the role that `time` plays for the others. That makes
    it the control for "does the Hamiltonian structure buy us anything at all?".
    """

    def __init__(self, num_qubits, depth=2, entangler="cx", rng=None):
        """
        Args:
            num_qubits (int): register size (n_in + n_mem).
            depth (int): number of entangler + rotation layers.
            entangler (str): "cx" for plain CNOTs, "cry" for controlled Ry with
                random angles.
            rng: seed or numpy Generator fixing the random angles.
        """
        super().__init__(num_qubits)
        if entangler not in ("cx", "cry"):
            raise ValueError(f"entangler must be 'cx' or 'cry', got {entangler!r}")

        self.name = f"random_{entangler}"
        self.depth = depth
        self.entangler = entangler

        rng = np.random.default_rng(rng)
        self.local_angles = rng.uniform(0, 2 * np.pi, (depth, num_qubits, 2))
        self.ctrl_angles = rng.uniform(0, 2 * np.pi, (depth, num_qubits))

    def _build(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        for d in range(self.depth):
            for i in range(self.num_qubits):
                target = (i + 1) % self.num_qubits
                if self.entangler == "cx":
                    qc.cx(i, target)
                else:
                    qc.cry(self.ctrl_angles[d, i], i, target)

            for i in range(self.num_qubits):
                qc.ry(self.local_angles[d, i, 0], i)
                qc.rz(self.local_angles[d, i, 1], i)

        return qc


class HamiltonianReservoir(Reservoir):
    """`1.` to `4.` Trotterised free evolution under a fixed Hamiltonian.

    The register evolves for a fixed `time` between encodings; on gate-based
    hardware that evolution is approximated by `steps` Trotter repetitions.
    """

    def __init__(self, hamiltonian, time=1.0, steps=1):
        super().__init__(hamiltonian.num_qubits)
        self.name = hamiltonian.name
        self.hamiltonian = hamiltonian
        self.time = time
        self.steps = steps

    def _build(self):
        return trotter.trotter_circuit(self.hamiltonian, self.time, self.steps)
