"""
SpinPulse backend: reproducibility and CLI wiring.

1. Same seed, same features and the reservoir always compiles to the same pulses
   (guards the seeded override of SpinPulse's compiler).
2. Every SpinPulse command-line option reaches the backend's configuration.
"""

import numpy as np
import pytest
from qiskit.quantum_info import PauliList

pytest.importorskip("spin_pulse")

import experiments.cli as cli
from qrc import pipeline
from qrc.backends.spinpulse import SpinPulseBackend, SpinPulseConfig
from qrc.encodings import AngleEncoding
from qrc.reservoirs import RandomCircuitReservoir

WINDOWS = np.random.default_rng(0).random((5, 2, 2))
OBSERVABLES = PauliList(["IIZ", "IZI", "ZII", "IZZ", "ZZI"])


def reservoir():
    # 3 qubits, 3 layers of random cx gates: couplings between non-neighbours need swaps
    # on SpinPulse's chain, which is where unseeded compilation would vary.
    return RandomCircuitReservoir(num_input_qubits=2, num_mem_qubits=1, depth=3, rng=1)


def run(**config):
    return SpinPulseBackend(SpinPulseConfig(**config)).run_batch(
        WINDOWS, AngleEncoding(2), reservoir(), OBSERVABLES
    )


# ------------------------------------------------------------------ 1. reproducibility


@pytest.mark.parametrize(
    "config",
    [
        dict(noise_type="pink", T2S=100.0, noise_samples=3),  # noise traces
        dict(noise_type="none", precision=0.05),  # shot sampling
    ],
    ids=["noise", "shots"],
)
def test_same_seed_gives_identical_features(config):
    first = run(seed=4, **config)
    np.testing.assert_array_equal(first, run(seed=4, **config))
    assert not np.allclose(first, run(seed=5, **config))


def test_reservoir_always_compiles_to_the_same_pulses():
    """Three separate backends must build the same pulse schedule and the same unitary.

    HardwareSpecs on its own compiles differently on every call (random routing and
    single-qubit merging); the backend rebuilds the compiler seeded to prevent that.
    """
    schedules, unitaries = [], []
    for _ in range(3):
        backend = SpinPulseBackend(SpinPulseConfig(noise_type="none"))
        timeline = backend._timeline_for(AngleEncoding(2), reservoir(), 2)
        schedules.append((timeline.res_duration, timeline.reservoir.n_layers))
        unitaries.append(backend._reservoir_unitary(timeline, None, 0))
    assert schedules[0] == schedules[1] == schedules[2]
    np.testing.assert_allclose(unitaries[0], unitaries[1], atol=1e-12)
    np.testing.assert_allclose(unitaries[0], unitaries[2], atol=1e-12)


# ------------------------------------------------------------------ 2. CLI wiring


def test_cli_options_reach_the_spinpulse_config():
    args = cli.build_parser().parse_args(
        [
            "qrc",
            "--backend", "spinpulse",
            "--noise", "quasistatic",
            "--t2", "500",
            "--tj", "200",
            "--only-idle",
            "--dd", "spin-echo",
            "--pulse-shape", "square",
            "--noise-samples", "7",
            "--precision", "0",
            "--seed", "5",
            "--reset-inputs",
        ]
    )
    config = pipeline.spinpulse_config(args)
    assert config == SpinPulseConfig(
        noise_type="quasistatic",
        T2S=500.0,
        TJS=200.0,
        only_idle=True,
        dynamical_decoupling="spin-echo",
        pulse_shape="square",
        noise_samples=7,
        precision=0.0,
        seed=5,
    )

    backend = pipeline.make_backend(
        args.backend, args.precision, args.seed, args.reset_inputs, config
    )
    assert isinstance(backend, SpinPulseBackend)
    assert backend.config is config
    assert backend.reset_inputs is True


@pytest.mark.parametrize(
    "bad",
    [dict(noise_type="pnik"), dict(dynamical_decoupling="echo"), dict(noise_samples=0), dict(precision=-0.1)],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        SpinPulseConfig(**bad)