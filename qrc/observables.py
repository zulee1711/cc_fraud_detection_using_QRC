from typing import List, Union
import itertools
from qiskit.quantum_info import Pauli, PauliList

def build_observables(specs: Union[str, List[str]], num_qubits: int) -> PauliList:
    """
    Dynamically builds Pauli observables for any combination of 1-body and 2-body bases.
    Supports composite string specifications split by '+' (eg: 'Z+XX', 'Y+XX-all', 'ZZ'):
    - Single-body : "X", "Y", "Z", "XYZ", "all-1body"
    - Two-body : "XX", "YY", "ZZ", "XYZ-2body"
    - Topology : "-all" for all pairs (eg. "XX-all")
                 "-nearest" for 1D chain correlations (default)

    Input : a specification string from the ones above

    Output : A Qiskit PauliList containing K Pauli strings, where K is the number of observables geenrated for the specification

    Args :
        specs : A string designating the desired observable operator combo
        num_qubits : the total number of qubits
    """
    if isinstance(specs, list):
        return PauliList(specs)

    # def _get_pauli_op(n, idx, label) -> Pauli:
    #     s = ['I'] * n
    #     s[-(idx + 1)] = label
    #     return Pauli("".join(s))

    # def _get_pauli_zz(n, idx1, idx2) -> Pauli:
    #     s = ['I'] * n
    #     s[-(idx1 + 1)] = 'Z'
    #     if s[-(idx2 + 1)] == 'Z':
    #         s[-(idx2 + 1)] = 'I'
    #     else:
    #         s[-(idx2 + 1)] = 'Z'
    #     return Pauli("".join(s))

    def _make_pauli(n : int, qubit_paili_map : dict) -> Pauli :
        """
        Cosntructs a Qiskit Pauli string
        """
        s = ["I"]*n
        for q_idx, p_label in qubit_paili_map.items():
            s[-(q_idx+1)] = p_label
        return Pauli("".join(s))

    # Parse specs string
    tokens = [t.strip() for t in specs.split("+")]
    observables = []

    for token in tokens:
        # Determine topology if present
        if token.endswith("-all"):
            topology = "all"
            base_token = token[:-4]
        elif token.endswith("-nearest"):
            topology = "nearest"
            base_token = token[:-8]
        else:
            topology = "nearest"  # default topology for 2-body terms
            base_token = token

        if base_token in ["all-1body", "XYZ"]:
            base_token = "X,Y,Z"
        elif base_token in ["all-2body", "XYZ-2body"]:
            base_token = "XX,YY,ZZ,XY,XZ,YX,YZ,ZX,ZY"

        # Single body operator expectations
        if base_token in ["X", "Y", "Z", "X,Y,Z"]:
                bases = ["X", "Y", "Z"] if base_token == "X,Y,Z" else [base_token]
                for i in range(num_qubits):
                    for b in bases:
                        observables.append(_make_pauli(num_qubits, {i: b}))

        # 2-body correlations expectations
        elif len(base_token) == 2 and base_token[0] in "XYZ" and base_token[1] in "XYZ":
                b1, b2 = base_token[0], base_token[1]
                pairs = (
                    list(itertools.combinations(range(num_qubits), 2))
                    if topology == "all"
                    else [(i, i + 1) for i in range(num_qubits - 1)]
                )
                for i, j in pairs:
                    observables.append(_make_pauli(num_qubits, {i: b1, j: b2}))

        # handles earlier expansion of base_token
        elif "," in base_token:
                    pairs = (
                        list(itertools.combinations(range(num_qubits), 2))
                        if topology == "all"
                        else [(i, i + 1) for i in range(num_qubits - 1)]
                    )
                    sub_bases = base_token.split(",")
                    for i, j in pairs:
                        for b_pair in sub_bases:
                            observables.append(
                                _make_pauli(num_qubits, {i: b_pair[0], j: b_pair[1]})
                            )
        else:
            raise ValueError(f"Unrecognized observable spec token: '{token}'")
    return PauliList(observables)

    # if specs=="Z":
    #     for i in range(num_qubits):
    #         observables.append(_get_pauli_op(num_qubits, i, "Z"))
    #     return observables

    # if specs=="Z+ZZ":
    #     for i in range(num_qubits):
    #         observables.append(_get_pauli_op(num_qubits, i, "Z"))
    #     for i in range(num_qubits-1):
    #         observables.append(_get_pauli_zz(num_qubits, i, i+1))
    #     return observables

    # raise ValueError("Invalid observables specification")