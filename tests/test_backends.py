import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qrc.backends.estimator import EstimatorBackend
from qrc.backends.numpy_backend import NumpyBackend
from qrc.encodings import AngleEncoding, DenseAngleEncoding, ReuploadingEncoding
from qrc.observables import build_observables
from qrc.hamiltonians import tfim_longitudinal
from qrc.reservoirs import HamiltonianReservoir, RandomCircuitReservoir

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
def test_sampling_noise_never_repeats(setup, sampling):
    """Identical windows get independent noise across PUBs and across calls (e.g. the
    train and test splits); Aer alone re-seeds every PUB and repeats its draws."""
    windows, encoder, reservoir, observables, _ = setup
    same = np.repeat(windows[:1], 2 * SAMPLES, axis=0)
    backend = EstimatorBackend(precision=0.05, sampling=sampling, seed=7, batch_size=SAMPLES)
    first = backend.run_batch(same, encoder, reservoir, observables)
    second = backend.run_batch(same, encoder, reservoir, observables)
    assert not np.allclose(first[:SAMPLES], first[SAMPLES:])
    assert not np.allclose(first, second)


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


# --------------------------------------------------------------------------- NumpyBackend

@pytest.mark.parametrize("reset_inputs", [False, True])
@pytest.mark.parametrize("encoding", ["angle", "angle-x", "dense", "reuploading"])
def test_numpy_backend_matches_reference(reset_inputs, encoding):
    reservoir = RandomCircuitReservoir(N_IN, N_MEM, rng=np.random.default_rng(0))
    n_features = N_IN
    if encoding == "angle":
        encoder = AngleEncoding(N_IN)
    elif encoding == "angle-x":
        encoder = AngleEncoding(N_IN, axis="x", scaling=2.0)
    elif encoding == "dense":
        encoder, n_features = DenseAngleEncoding(N_IN), N_IN + 1
    else:
        encoder = ReuploadingEncoding(N_IN, num_layers=3, rng=1)
    observables = build_observables("XYZ+XYZ-2body", reservoir.num_qubits)
    windows = np.random.default_rng(2).uniform(0, 1, (SAMPLES, WINDOW_LENGTH, n_features))
    exact = exact_features(windows, encoder, reservoir, observables, reset_inputs=reset_inputs)
    result = NumpyBackend(reset_inputs=reset_inputs).run_batch(windows, encoder, reservoir, observables)
    np.testing.assert_allclose(result, exact, atol=1e-10)


def test_numpy_backend_repeats_features_like_estimator():
    """One feature on three input qubits: both backends repeat it cyclically."""
    reservoir = RandomCircuitReservoir(N_IN, N_MEM, rng=np.random.default_rng(0))
    encoder, observables = AngleEncoding(N_IN), build_observables("Z+XX", reservoir.num_qubits)
    windows = np.random.default_rng(3).uniform(0, 1, (4, WINDOW_LENGTH, 1))
    expected = EstimatorBackend(reset_inputs=True).run_batch(windows, encoder, reservoir, observables)
    result = NumpyBackend(reset_inputs=True).run_batch(windows, encoder, reservoir, observables)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_numpy_backend_matches_estimator_on_hamiltonian_reservoir():
    reservoir = HamiltonianReservoir(tfim_longitudinal(4, rng=0), 1, time=2.0, steps=3)
    encoder, observables = AngleEncoding(1), build_observables("XYZ+XYZ-2body", 4)
    windows = np.random.default_rng(4).uniform(0, 1, (6, 5, 1))
    for reset_inputs in (False, True):
        expected = EstimatorBackend(reset_inputs=reset_inputs).run_batch(windows, encoder, reservoir, observables)
        result = NumpyBackend(reset_inputs=reset_inputs).run_batch(windows, encoder, reservoir, observables)
        np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.parametrize("reset_inputs", [False, True])
def test_numpy_backend_chunking_does_not_change_results(setup, reset_inputs):
    windows, encoder, reservoir, observables, _ = setup
    whole = NumpyBackend(reset_inputs=reset_inputs).run_batch(windows, encoder, reservoir, observables)
    # a budget below one window's states forces chunks of a single window
    chunked = NumpyBackend(reset_inputs=reset_inputs, max_chunk_bytes=1).run_batch(
        windows, encoder, reservoir, observables)
    np.testing.assert_allclose(chunked, whole, atol=1e-12)


def test_numpy_backend_empty_input_and_invalid_arguments(setup):
    windows, encoder, reservoir, observables, _ = setup
    assert NumpyBackend().run_batch(windows[:0], encoder, reservoir, observables).shape == (0, len(observables))
    with pytest.raises(ValueError):
        NumpyBackend(max_chunk_bytes=0)
    with pytest.raises(ValueError):
        NumpyBackend(precision=-0.1)


@pytest.mark.parametrize("reset_inputs", [False, True])
def test_numpy_gaussian_matches_estimator_gaussian(reset_inputs):
    """Same exact values and the same noise draws, call after call (one call per split)."""
    reservoir = RandomCircuitReservoir(1, 3, rng=0)
    encoder, observables = AngleEncoding(1), build_observables("XYZ+XYZ-2body", 4)
    rng = np.random.default_rng(5)
    splits = [rng.uniform(0, 1, (n, WINDOW_LENGTH, 1)) for n in (9, 4, 6)]
    aer = EstimatorBackend(precision=0.01, seed=7, sampling="gaussian", reset_inputs=reset_inputs)
    numpy = NumpyBackend(reset_inputs=reset_inputs, precision=0.01, seed=7)
    for windows in splits:
        expected = aer.run_batch(windows, encoder, reservoir, observables)
        result = numpy.run_batch(windows, encoder, reservoir, observables)
        np.testing.assert_allclose(result, expected, atol=1e-10)
    # the noise is really there
    exact = NumpyBackend(reset_inputs=reset_inputs).run_batch(splits[0], encoder, reservoir, observables)
    noisy = NumpyBackend(reset_inputs=reset_inputs, precision=0.01, seed=7).run_batch(
        splits[0], encoder, reservoir, observables)
    assert 0.005 < np.std(noisy - exact) < 0.02
