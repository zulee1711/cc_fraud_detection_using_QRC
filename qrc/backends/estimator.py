from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import PauliList
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qrc.backends.base import Backend
from qrc.encodings import Encoding
from qrc.reservoirs import Reservoir

SAMPLING_MODES = ("gaussian", "shots")


class EstimatorBackend(Backend):
    """Runs the QRC circuits through a Qiskit Estimator primitive.

    The circuit is built once, with the window's features as a ``ParameterVector``,
    and each window (one sample: ``window_length`` consecutive transactions) is one set
    of its parameter values. The samples are split into PUBs of ``batch_size`` windows
    each, all submitted in a single job.

    Args:
        precision (float): target standard error of every estimated expectation value.
            For Pauli observables, ``precision ~ 1 / sqrt(shots)``. ``0.0`` means exact
            (only allowed with ``sampling="gaussian"``).
        noise_model (NoiseModel, optional): Aer noise model applied to the simulation.
        seed (int, optional): seed, for reproducible results.
        sampling (str): how ``precision`` is realised.

            * ``"gaussian"``: Aer's ``EstimatorV2``. Computes the exact expectation values
              of the (noisy) circuit and adds Gaussian noise of width ``precision``.
              Fast, but no measurement is actually sampled.
            * ``"shots"``: Qiskit's ``BackendEstimatorV2`` on ``AerSimulator``. Appends
              the measurement bases, samples ``~1/precision**2`` shots per circuit and
              averages the counts, as on hardware. Slower, but a genuine shot simulation.
        batch_size (int): number of samples (whole windows) per PUB. Windows are never
            split across PUBs. Aer slows down on very large PUBs, so large datasets are
            split into several PUBs, all submitted in a single job.
        reset_inputs (bool): reset the input qubits to |0> before each encoding step after
            the first, so each transaction overwrites the input register instead of rotating
            on top of the previous state. Memory qubits are never reset. Reset is not unitary,
            so this switches Aer to its density-matrix method (exact, but memory grows as 4**n).
    """

    def __init__(self, precision: float = 0.0, noise_model: Optional[NoiseModel] = None,
                 seed: Optional[int] = None, sampling: str = "gaussian", batch_size: int = 250,
                 reset_inputs: bool = False):
        if sampling not in SAMPLING_MODES:
            raise ValueError(f"sampling must be one of {SAMPLING_MODES}, got {sampling!r}")
        if precision < 0:
            raise ValueError("precision must be non-negative")
        if sampling == "shots" and precision == 0:
            raise ValueError("sampling='shots' needs precision > 0 (number of shots ~ 1 / precision**2)")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.batch_size = batch_size
        self.precision = precision
        self.noise_model = noise_model
        self.seed = seed
        self.sampling = sampling
        self.reset_inputs = reset_inputs

        backend_options = {}
        if noise_model is not None:
            backend_options["noise_model"] = noise_model
        if reset_inputs:
            backend_options["method"] = "density_matrix"
        if sampling == "gaussian":
            self._backend = None
            run_options = {} if seed is None else {"seed_simulator": seed}
            self.estimator = AerEstimator(options={
                "default_precision": precision,
                "backend_options": backend_options,
                "run_options": run_options,
            })
        else:
            if seed is not None:
                backend_options["seed_simulator"] = seed
            self._backend = AerSimulator(**backend_options)
            self.estimator = BackendEstimatorV2(backend=self._backend,
                                                options={"default_precision": precision})

    def _build_circuit(self, encoder: Encoding, reservoir: Reservoir, window_length: int) -> QuantumCircuit:
        """Build the parametrized circuit for a window of ``window_length`` transactions.
        Has ``window_length * num_input_qubits`` parameters, ordered transaction by transaction."""
        n_in = reservoir.num_input_qubits
        theta = ParameterVector("x", window_length * n_in)
        reservoir_circuit = reservoir.circuit()
        circuit = QuantumCircuit(reservoir.num_qubits)
        for t in range(window_length):
            # 1. discard and re-prepare the input qubits (already |0> at the first step)
            if self.reset_inputs and t > 0:
                circuit.reset(range(n_in))
            # 2. encode into the input qubits
            circuit.compose(encoder.encode(theta[t * n_in:(t + 1) * n_in]),
                            qubits=range(n_in), inplace=True)
            # 3. evolve the whole register
            circuit.compose(reservoir_circuit, inplace=True)
        return circuit

    def run_batch(self, windows: np.ndarray, encoder: Encoding, reservoir: Reservoir, observables: PauliList) -> np.ndarray:
        """Run all windows in one job, ``batch_size`` windows per PUB. Observables are measured only once, after the last transaction.
        Returns np array with shape ``(samples, observables)``."""
        samples, window_length, num_features = windows.shape
        if samples == 0:
            return np.empty((0, len(observables)))
        n_in = reservoir.num_input_qubits
        circuit = self._build_circuit(encoder, reservoir, window_length)
        if self._backend is not None:
            circuit = transpile(circuit, self._backend, optimization_level=0)

        # tile/truncate the features onto the input qubits
        resized = windows[..., np.arange(n_in) % num_features]
        # (samples, 1, parameters): the singleton axis broadcasts against the observables
        parameter_values = resized.astype(float).reshape(samples, 1, window_length * n_in)

        pubs = [(circuit, observables, parameter_values[start:start + self.batch_size])
                for start in range(0, samples, self.batch_size)]
        job = self.estimator.run(pubs)
        return np.concatenate([np.asarray(result.data.evs) for result in job.result()])
