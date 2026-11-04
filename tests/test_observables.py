from qiskit.quantum_info import Pauli

from qrc.observables import build_observables


def test_xyz_strings_are_every_single_letter_pauli_string():
    n = 4
    observables = build_observables("XYZ-strings", n)
    # experiments/fraud_2's construction, whose order the readout features follow
    expected = [
        "".join(letter if (m >> (n - 1 - q)) & 1 else "I" for q in range(n))
        for letter in "XYZ"
        for m in range(1, 2**n)
    ]

    assert [p.to_label() for p in observables] == expected
    assert len(observables.group_qubit_wise_commuting()) == 3


def test_xyz_strings_combine_with_other_tokens():
    observables = build_observables("Z+XYZ-strings", 2)
    assert len(observables) == 2 + 9 and observables[0] == Pauli("IZ")
