import numpy as np
import pytest
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qrc.backends.estimator import EstimatorBackend
from qrc.backends.statevector import StatevectorBackend
from qrc.encodings import AngleEncoding
from qrc.observables import build_observables
from qrc.reservoirs import RandomCircuitReservoir

N_IN, N_MEM, WINDOW_LENGTH, SAMPLES = 3, 2, 4, 12


@pytest.fixture
def setup():
    reservoir = RandomCircuitReservoir(N_IN, N_MEM, rng=np.random.default_rng(0))
    encoder = AngleEncoding(N_IN)
    observables = build_observables("Z+XX", N_IN + N_MEM)
    windows = np.random.default_rng(1).uniform(0, 1, (SAMPLES, WINDOW_LENGTH, N_IN))
    exact = StatevectorBackend().run_batch(windows, encoder, reservoir, observables)
    return windows, encoder, reservoir, observables, exact


def run(backend, setup):
    windows, encoder, reservoir, observables, _ = setup
    return backend.run_batch(windows, encoder, reservoir, observables)


def depolarizing_noise_model():
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ["ry", "rx", "rz", "h", "sx", "x"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
    return noise_model


def test_exact_estimator_matches_statevector(setup):
    result = run(EstimatorBackend(), setup)
    assert result.shape == setup[-1].shape
    np.testing.assert_allclose(result, setup[-1], atol=1e-10)


@pytest.mark.parametrize("sampling", ["gaussian", "shots"])
def test_finite_precision_is_close_to_exact(setup, sampling):
    precision = 0.05
    result = run(EstimatorBackend(precision=precision, sampling=sampling, seed=3), setup)
    error = np.abs(result - setup[-1])
    assert result.shape == setup[-1].shape
    assert error.max() > 0  # actually noisy
    assert error.mean() < precision  # expected mean |error| is ~0.8 * precision


@pytest.mark.parametrize("sampling", ["gaussian", "shots"])
def test_seed_is_reproducible(setup, sampling):
    first = run(EstimatorBackend(precision=0.05, sampling=sampling, seed=7), setup)
    second = run(EstimatorBackend(precision=0.05, sampling=sampling, seed=7), setup)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("sampling", ["gaussian", "shots"])
def test_noise_model_changes_the_features(setup, sampling):
    noisy = run(EstimatorBackend(precision=0.02 if sampling == "shots" else 0.0, sampling=sampling,
                                 noise_model=depolarizing_noise_model(), seed=1), setup)
    assert np.abs(noisy - setup[-1]).mean() > 0.02


@pytest.mark.parametrize("batch_size", [1, 5, SAMPLES])
def test_batching_matches_reference(setup, batch_size):
    """Splitting the windows into PUBs (5 leaves a short last one) changes nothing."""
    result = run(EstimatorBackend(batch_size=batch_size), setup)
    assert result.shape == setup[-1].shape
    np.testing.assert_allclose(result, setup[-1], atol=1e-10)


def test_empty_input(setup):
    windows, encoder, reservoir, observables, _ = setup
    result = EstimatorBackend().run_batch(windows[:0], encoder, reservoir, observables)
    assert result.shape == (0, len(observables))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        EstimatorBackend(sampling="bogus")
    with pytest.raises(ValueError):
        EstimatorBackend(precision=-0.1)
    with pytest.raises(ValueError):
        EstimatorBackend(precision=0.0, sampling="shots")
    with pytest.raises(ValueError):
        EstimatorBackend(batch_size=0)
