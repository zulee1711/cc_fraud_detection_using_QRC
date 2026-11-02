"""The injection points of qrc.pipeline: an explicit backend and Hamiltonian builder."""

import numpy as np
import pytest

import experiments.cli as cli
from qrc import pipeline
from qrc.backends.numpy_backend import NumpyBackend
from qrc.hamiltonians import tfim


def _data(n_features=1):
    rng = np.random.default_rng(0)
    return {f"X_{split}": rng.uniform(size=(3, 2, n_features)) for split in ("train", "validation", "test")}


def _qrc_args(*extra):
    return cli.build_parser().parse_args(["qrc", "--feature-set", "1", "--n-mem-qubits", "2", *extra])


def test_backend_argument_is_used_instead_of_args_backend():
    backend = NumpyBackend()
    protocol, _ = pipeline.build_protocol(_qrc_args("--backend", "estimator-exact"), backend=backend)
    assert protocol.backend is backend


def test_hamiltonian_argument_replaces_the_builder():
    calls = []

    def builder(num_qubits, topology, rng):
        calls.append((num_qubits, topology, rng))
        return tfim(num_qubits, topology=topology, rng=rng)

    args = _qrc_args("--hamiltonian", "tfim", "--backend", "numpy")
    features, _ = pipeline.qrc_features(_data(), args, hamiltonian=builder)
    reference, _ = pipeline.qrc_features(_data(), args)

    assert len(calls) == 1 and calls[0][0] == 3 and calls[0][2] == args.seed
    np.testing.assert_allclose(features["test"], reference["test"])
    assert pipeline.HAMILTONIAN_BUILDERS["tfim"] is tfim  # module state untouched


def test_hamiltonian_argument_rejected_for_random_reservoir():
    with pytest.raises(ValueError, match="random-circuit"):
        pipeline.build_protocol(_qrc_args("--hamiltonian", "random"), hamiltonian=tfim)


def test_field_option_matches_a_field_scaled_builder():
    def builder(num_qubits, topology, rng):
        return tfim(num_qubits, topology=topology, rng=rng, field=0.3)

    args = _qrc_args("--hamiltonian", "tfim", "--backend", "numpy")
    reference, _ = pipeline.qrc_features(_data(), args, hamiltonian=builder)
    features, info = pipeline.qrc_features(_data(), _qrc_args("--hamiltonian", "tfim", "--backend", "numpy", "--field", "0.3"))

    np.testing.assert_allclose(features["test"], reference["test"])
    assert info["field"] == 0.3


def test_field_option_rejected_with_a_hamiltonian_argument():
    with pytest.raises(ValueError, match="--field"):
        pipeline.build_protocol(_qrc_args("--hamiltonian", "tfim", "--field", "0.5"), hamiltonian=tfim)


def test_field_option_rejected_with_the_random_reservoir():
    with pytest.raises(ValueError, match="--field"):
        pipeline.build_protocol(_qrc_args("--hamiltonian", "random", "--field", "0.5"))
