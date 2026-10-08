import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator

from qrc.backends.estimator import EstimatorBackend
from qrc.backends.statevector import StatevectorBackend
from qrc.encodings import AngleEncoding, DenseAngleEncoding, ReuploadingEncoding
from qrc.observables import build_observables
from qrc.reservoirs import RandomCircuitReservoir

N_IN = 3
VALUES = [0.1, 0.5, 0.9]


def test_single_layer_is_angle_encoding():
    reuploading = ReuploadingEncoding(N_IN, num_layers=1).encode(VALUES)
    angle = AngleEncoding(N_IN).encode(VALUES)
    assert Operator(reuploading).equiv(Operator(angle))


def test_layers_do_not_collapse_to_one_rescaled_layer():
    reuploading = ReuploadingEncoding(N_IN, num_layers=2, rng=0).encode(VALUES)
    rescaled = AngleEncoding(N_IN, scaling=2 * np.pi).encode(VALUES)
    assert not Operator(reuploading).equiv(Operator(rescaled))


def test_interleaved_angles_are_fixed_by_rng():
    a = ReuploadingEncoding(N_IN, num_layers=3, rng=7)
    b = ReuploadingEncoding(N_IN, num_layers=3, rng=7)
    assert a.interleave_angles.shape == (2, N_IN)
    assert Operator(a.encode(VALUES)).equiv(Operator(b.encode(VALUES)))


def test_scaling_is_set_on_every_layer():
    encoding = ReuploadingEncoding(N_IN, num_layers=3)
    encoding.set_scaling_based_on_input_range(-1.0, 1.0)
    assert all(layer.scaling == pytest.approx(np.pi / 2) for layer in encoding.layers)


@pytest.mark.parametrize("kwargs", [dict(num_layers=0), dict(axis="z", interleave_axis="z")])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ReuploadingEncoding(N_IN, **kwargs)


def test_accepts_parameters():
    circuit = ReuploadingEncoding(N_IN, num_layers=2).encode(ParameterVector("x", N_IN))
    assert circuit.num_parameters == N_IN


def test_dense_encoding_applies_two_features_per_qubit():
    encoding = DenseAngleEncoding(2)
    circuit = encoding.encode([0.1, 0.2, 0.3, 0.4])

    expected = QuantumCircuit(2)
    expected.ry(0.1 * np.pi, 0)
    expected.rz(0.3 * np.pi, 0)
    expected.ry(0.2 * np.pi, 1)
    expected.rz(0.4 * np.pi, 1)
    assert Operator(circuit).equiv(Operator(expected))


def test_dense_encoding_accepts_a_partial_second_feature_layer():
    encoding = DenseAngleEncoding(2)
    circuit = encoding.encode([0.1, 0.2, 0.3])

    expected = QuantumCircuit(2)
    expected.ry(0.1 * np.pi, 0)
    expected.rz(0.3 * np.pi, 0)
    expected.ry(0.2 * np.pi, 1)
    assert Operator(circuit).equiv(Operator(expected))


def test_dense_encoding_uses_scaling_and_min_value():
    encoding = DenseAngleEncoding(2, axis1="x", axis2="y", scaling=2)
    circuit = encoding.encode([0.5, 1.0, 1.5], min_value=0.5)

    expected = QuantumCircuit(2)
    expected.rx(0, 0)
    expected.ry(2, 0)
    expected.rx(1, 1)
    assert Operator(circuit).equiv(Operator(expected))


@pytest.mark.parametrize("values", [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]])
def test_dense_encoding_rejects_input_outside_supported_range(values):
    with pytest.raises(ValueError):
        DenseAngleEncoding(2).encode(values)


def test_dense_encoding_rejects_identical_axes():
    with pytest.raises(ValueError):
        DenseAngleEncoding(2, axis1="y", axis2="y")


def test_estimator_matches_statevector():
    reservoir = RandomCircuitReservoir(N_IN, 2, rng=np.random.default_rng(0))
    encoder = ReuploadingEncoding(N_IN, num_layers=3, rng=1)
    observables = build_observables("Z+XX", reservoir.num_qubits)
    windows = np.random.default_rng(2).uniform(0, 1, (5, 2, N_IN))
    exact = StatevectorBackend().run_batch(windows, encoder, reservoir, observables)
    estimated = EstimatorBackend().run_batch(windows, encoder, reservoir, observables)
    np.testing.assert_allclose(estimated, exact, atol=1e-8)
