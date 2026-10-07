import numpy as np
import pytest
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator

from qrc.backends.estimator import EstimatorBackend
from qrc.backends.statevector import StatevectorBackend
from qrc.encodings import AngleEncoding, ReuploadingEncoding
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


def test_estimator_matches_statevector():
    reservoir = RandomCircuitReservoir(N_IN, 2, rng=np.random.default_rng(0))
    encoder = ReuploadingEncoding(N_IN, num_layers=3, rng=1)
    observables = build_observables("Z+XX", reservoir.num_qubits)
    windows = np.random.default_rng(2).uniform(0, 1, (5, 2, N_IN))
    exact = StatevectorBackend().run_batch(windows, encoder, reservoir, observables)
    estimated = EstimatorBackend().run_batch(windows, encoder, reservoir, observables)
    np.testing.assert_allclose(estimated, exact, atol=1e-8)
