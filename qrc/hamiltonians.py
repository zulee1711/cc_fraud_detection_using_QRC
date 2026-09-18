from qiskit.quantum_info import SparsePauliOp
import numpy as np

class Hamiltonian:
    def __init__(self, num_qubits: int, ops: SparsePauliOp, seed: int):
        np.random.default_rng(seed)
        self.num_qubits = num_qubits
        self.ops = ops
        self.seed = seed

def build_disordered_tfim(n, J, h ,W) -> Hamiltonian:
    """Builds a disordered transverse-field Ising Hamiltonian, of the form
                  Σ J_ij Z_i Z_j  +   Σ h_i Z_i +   Σ g X_i
     with J_ij ~ U[J/2, J], h_i ~ U[h-W, h+W] and g ~ J
     """
    ops_list = []
    J_couplings = np.random.uniform(J / 2, J, n - 1)
    h_disorders = np.random.uniform(h-W, h+W, n)
    g = J
    for i in range(n - 1):
        ops_list.append(("ZZ", [i, i + 1], J_couplings[i]))
    for i in range(n):
        ops_list.append(("Z", [i], h_disorders[i]))
    for i in range(n):
        ops_list.append(("X", [i], g))

    ops = SparsePauliOp.from_sparse_list(ops_list, num_qubits=n)
    return Hamiltonian(n, ops, 42)

# H = build_disordered_tfim(5, 0, 0, 0)
