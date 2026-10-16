import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qrc.backends.estimator import EstimatorBackend
from qrc.encodings import AngleEncoding, DenseAngleEncoding, ReuploadingEncoding
from qrc.observables import build_observables
from qrc.reservoirs import RandomCircuitReservoir

N_IN, N_MEM, WINDOW_LENGTH, SAMPLES = 3, 2, 4, 12


def exact_features(windows, encoder, reservoir, observables, reset_inputs=False):
    """Reference implementation the backends are tested against: one window at a
    time, as an exact density matrix, step by step in plain ``qiskit.quantum_info``.

    Returns np array with shape ``(samples, observables)``.
    """
    n_in = reservoir.num_input_qubits
    features = []
    for window in windows:
        rho = DensityMatrix.from_label("0" * reservoir.num_qubits)
        for t, values in enumerate(window):
            if reset_inputs and t > 0:
                rho = rho.reset(list(range(n_in)))
            # same cyclic repetition as the backends when the encoding takes more values than features
            input_width = encoder.input_width(values.size)
            step = QuantumCircuit(reservoir.num_qubits)
            step.compose(encoder.encode(values[np.arange(input_width) % values.size]),
                         qubits=range(n_in), inplace=True)
            step.compose(reservoir.circuit(), inplace=True)
            rho = rho.evolve(step)
        features.append([rho.expectation_value(o).real for o in observables])
    return np.asarray(features)



@pytest.fixture
def setup():
    reservoir = RandomCircuitReservoir(N_IN, N_MEM, rng=np.random.default_rng(0))
    encoder = AngleEncoding(N_IN)
    observables = build_observables("Z+XX", N_IN + N_MEM)
    windows = np.random.default_rng(1).uniform(0, 1, (SAMPLES, WINDOW_LENGTH, N_IN))
    exact = exact_features(windows, encoder, reservoir, observables)
    return windows, encoder, reservoir, observables, exact


def run(backend, setup):
    windows, encoder, reservoir, observables, _ = setup
    return backend.run_batch(windows, encoder, reservoir, observables)


def depolarizing_noise_model():
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ["ry", "rx", "rz", "h", "sx", "x"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
    return noise_model


def test_exact_estimator_matches_reference(setup):
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


def test_reset_inputs_matches_reference(setup):
    """With reset, each step re-prepares the input qubits in |0> before encoding; memory qubits keep their state."""
    windows, encoder, reservoir, observables, no_reset = setup
    expected = exact_features(windows, encoder, reservoir, observables, reset_inputs=True)
    result = run(EstimatorBackend(reset_inputs=True), setup)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert np.abs(result - no_reset).max() > 0.01  # the reset actually changes the features


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

def test_estimator_backend_supports_dense_encoding():
    n_input_qubits = 3
    n_features = 4

    encoder = DenseAngleEncoding(n_input_qubits)
    reservoir = RandomCircuitReservoir(
        n_input_qubits,
        N_MEM,
        rng=np.random.default_rng(0),
    )
    observables = build_observables("Z+XX", n_input_qubits + N_MEM)
    windows = np.random.default_rng(2).uniform(
        0,
        1,
        (2, WINDOW_LENGTH, n_features),
    )

    result = EstimatorBackend().run_batch(
        windows,
        encoder,
        reservoir,
        observables,
    )

    assert result.shape == (2, len(observables))


def test_estimator_matches_reference():
    reservoir = RandomCircuitReservoir(N_IN, 2, rng=np.random.default_rng(0))
    encoder = ReuploadingEncoding(N_IN, num_layers=3, rng=1)
    observables = build_observables("Z+XX", reservoir.num_qubits)
    windows = np.random.default_rng(2).uniform(0, 1, (5, 2, N_IN))
    exact = exact_features(windows, encoder, reservoir, observables)
    estimated = EstimatorBackend().run_batch(windows, encoder, reservoir, observables)
    np.testing.assert_allclose(estimated, exact, atol=1e-8)
