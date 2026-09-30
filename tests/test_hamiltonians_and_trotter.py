import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from scipy.linalg import expm

from qrc.hamiltonians import (
    Hamiltonian,
    native_exchange,
    tfim,
    tfim_longitudinal,
    tfim_native,
)
from qrc.trotter import pauli_rotation, trotter_circuit


@pytest.fixture
def simple_hamiltonian():
    ops = SparsePauliOp.from_sparse_list(
        [
            ("Z", [0], 0.7),
            ("X", [1], -0.5),
            ("ZZ", [0, 1], 0.2),
        ],
        2,
    )
    return Hamiltonian("toy", 2, ops, {"z0": 0.7, "x1": -0.5, "zz01": 0.2})


def test_hamiltonian_repr_and_metadata():
    ops = SparsePauliOp.from_sparse_list([("X", [0], 1.0), ("Y", [1], 2.0)], 2)
    ham = Hamiltonian("demo", 2, ops, {"alpha": 1.0})

    assert ham.name == "demo"
    assert ham.num_qubits == 2
    assert len(ham.ops) == 2
    assert ham.params == {"alpha": 1.0}
    assert "Hamiltonian(" in repr(ham)


@pytest.mark.parametrize(
    ("builder", "expected_terms"),
    [
        (tfim, 2 * 3 - 1),
        (tfim_longitudinal, 3 * 3 - 1),
        (tfim_native, 3 * 3 - 1),
        (native_exchange, 5 * 3 - 3),
    ],
)
def test_hamiltonian_builders_create_valid_operators(builder, expected_terms):
    h = builder(3, v=2.0)

    assert isinstance(h, Hamiltonian)
    assert h.num_qubits == 3
    assert h.ops.num_qubits == 3
    assert len(h.ops.paulis) == expected_terms
    assert h.name
    assert isinstance(h.params, dict)


def test_tfim_specific_structure_and_randomized_params():
    h = tfim(4, v=1.5)

    assert h.name == "tfim-textbook"
    assert len(h.params) == 3
    assert all(key.startswith("J") for key in h.params)
    assert all(0.0 <= value <= 1.5 for value in h.params.values())

    h_long = tfim_longitudinal(4, v=1.5)
    assert h_long.name == "tfim-longitudinal"
    assert len(h_long.params) == 7
    assert any(key.startswith("g") for key in h_long.params)

    h_native = tfim_native(4, v=1.5)
    assert h_native.name == "tfim-native"
    assert len(h_native.params) == 7

    h_exchange = native_exchange(4, v=1.5, eta=0.7)
    assert h_exchange.name == "tfim-native"
    assert len(h_exchange.params) == 10
    assert all(key.startswith(("J", "g", "etaJ")) for key in h_exchange.params)
    assert all(np.isfinite(value) for value in h_exchange.params.values())


@pytest.mark.parametrize(
    ("pauli_label", "expected_gate_names"),
    [
        ("I", []),
        ("X", ["rx"]),
        ("Y", ["ry"]),
        ("Z", ["rz"]),
        ("ZZ", ["rzz"]),
        ("XX", ["ry", "rzz"]),
        ("YY", ["rx", "rzz"]),
    ],
)
def test_pauli_rotation_adds_expected_gates(pauli_label, expected_gate_names):
    qc = QuantumCircuit(2 if len(pauli_label) > 1 else 1)
    pauli = Pauli(pauli_label)

    pauli_rotation(qc, pauli, 0.37)

    ops = qc.count_ops()
    if not expected_gate_names:
        assert ops == {}
        return

    for gate_name in expected_gate_names:
        assert gate_name in ops


def test_pauli_rotation_rejects_unsupported_terms():
    qc = QuantumCircuit(2)

    with pytest.raises(ValueError, match="Unsupported Pauli term"):
        pauli_rotation(qc, Pauli("XZ"), 0.5)

    with pytest.raises(ValueError, match="Only 1- and 2-qubit Pauli terms supported"):
        pauli_rotation(qc, Pauli("XYZ"), 0.5)


def test_trotter_circuit_decomposed_matches_expected_structure(simple_hamiltonian):
    qc = trotter_circuit(simple_hamiltonian, time=0.8, steps=3, decompose=True)

    assert qc.num_qubits == 2
    assert len(qc.data) > 0
    assert any(name in qc.count_ops() for name in ["rx", "ry", "rz", "rzz"])
    assert qc.depth() > 0


def test_trotter_circuit_matrix_form_matches_exact_product(simple_hamiltonian):
    time = 0.6
    steps = 2
    dt = time / steps

    qc = trotter_circuit(simple_hamiltonian, time=time, steps=steps, decompose=False)
    matrix = Operator(qc).data

    expected = np.eye(4, dtype=complex)
    for pauli, coeff in zip(
        simple_hamiltonian.ops.paulis, simple_hamiltonian.ops.coeffs
    ):
        U_j = expm(-1j * coeff * pauli.to_matrix() * dt)
        expected = U_j @ expected
    expected = np.linalg.matrix_power(expected, steps)

    assert np.allclose(matrix, expected)
    np.testing.assert_allclose(
        matrix.conj().T @ matrix,
        np.eye(4),
        atol=1e-12,
        rtol=1e-12,
    )


def test_trotter_decomposed_circuit_matches_unitary_version(simple_hamiltonian):
    time = 0.8
    steps = 3

    decomposed = trotter_circuit(
        simple_hamiltonian, time=time, steps=steps, decompose=True
    )
    unitary = trotter_circuit(
        simple_hamiltonian, time=time, steps=steps, decompose=False
    )

    decomposed_matrix = Operator(decomposed).data
    unitary_matrix = Operator(unitary).data

    np.testing.assert_allclose(
        decomposed_matrix, unitary_matrix, atol=1e-12, rtol=1e-12
    )


def test_tfim_trotter_decomposed_matches_unitary_version():
    h = tfim(3, v=1.2)
    time = 0.7
    steps = 4

    decomposed = trotter_circuit(h, time=time, steps=steps, decompose=True)
    unitary = trotter_circuit(h, time=time, steps=steps, decompose=False)

    np.testing.assert_allclose(
        Operator(decomposed).data,
        Operator(unitary).data,
        atol=1e-12,
        rtol=1e-12,
    )


def test_trotter_circuit_zero_time_is_identity():
    h = tfim(2, v=1.0)
    qc = trotter_circuit(h, time=0.0, steps=5, decompose=True)
    qc_unitary = trotter_circuit(h, time=0.0, steps=5, decompose=False)

    assert qc.num_qubits == 2
    np.testing.assert_allclose(Operator(qc).data, np.eye(4))
    np.testing.assert_allclose(Operator(qc_unitary).data, np.eye(4))
