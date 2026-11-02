"""
Test for per-split noise and chunked windows in the backend. Test the Arm B script in SpinPulse end to end.
"""

import csv

import numpy as np
import pytest
from qiskit.quantum_info import PauliList

pytest.importorskip("spin_pulse")

from experiments.spinpulse_armb import main
from qrc.backends.spinpulse import SpinPulseBackend, SpinPulseConfig
from qrc.encodings import AngleEncoding
from qrc.hamiltonians import tfim
from qrc.reservoirs import HamiltonianReservoir

WINDOWS = np.random.default_rng(0).random((7, 3, 1))
OBSERVABLES = PauliList(["IZ", "ZI", "ZZ", "XI"])


def reservoir():
    return HamiltonianReservoir(tfim(2, rng=1), 1, steps=2)


def run(backend):
    return backend.run_batch(WINDOWS, AngleEncoding(1), reservoir(), OBSERVABLES)


def test_every_call_sees_the_same_chip_and_draws_fresh_shots():
    """Train, validation and test (successive calls) see the same noise realizations, so the
    exact features of the same windows are identical; only the shots are drawn afresh. A new
    backend with the same seed repeats the same sequence."""
    config = SpinPulseConfig(noise_type="pink", T2S=50.0, noise_samples=3, seed=3)
    backend = SpinPulseBackend(config)
    np.testing.assert_array_equal(run(backend), run(backend))

    observables = PauliList(["IZ", "ZI", "ZZ"])
    shots = SpinPulseBackend(SpinPulseConfig(noise_type="pink", T2S=50.0, noise_samples=3, precision=0.1, seed=3))
    first, second = (shots.run_batch(WINDOWS, AngleEncoding(1), reservoir(), observables) for _ in range(2))
    assert not np.allclose(first, second)
    again = SpinPulseBackend(shots.config)
    np.testing.assert_array_equal(first, again.run_batch(WINDOWS, AngleEncoding(1), reservoir(), observables))


@pytest.mark.parametrize("reset", [False, True])
def test_chunked_windows_give_identical_features(reset):
    """Processing the windows one at a time must not change anything."""
    settings = dict(noise_type="pink", T2S=50.0, noise_samples=3, seed=3)
    whole = run(SpinPulseBackend(SpinPulseConfig(**settings), reset_inputs=reset))
    one_by_one = run(SpinPulseBackend(SpinPulseConfig(max_chunk_bytes=1, **settings), reset_inputs=reset))
    np.testing.assert_allclose(one_by_one, whole, atol=1e-12)


def test_armb_script_writes_scores_and_reuses_cached_features(tmp_path):
    out, cache = tmp_path / "armb.csv", tmp_path / "features"
    argv = [
        "--levels", "ideal", "sp-ideal", "1", "--seeds", "42", "--precisions", "0", "0.05",
        "--noise-samples", "2", "--limit", "150", "--out", str(out), "--cache-dir", str(cache),
        "qrc", "--size", "small", "--feature-set", "1", "--window-length", "2",
        "--n-mem-qubits", "1", "--depth", "1", "--reset-inputs", "--observables", "Z+ZZ",
    ]
    main(argv)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    # logistic: 2 precisions x 2 splits; qrc: 3 levels x 2 precisions x 2 splits
    assert len(rows) == 4 + 12
    assert all(0.0 <= float(r["average_precision"]) <= 1.0 for r in rows)
    assert all(float(r["window_us"]) > 0 for r in rows if r["level"] in ("sp-ideal", "1"))
    assert len(list(cache.glob("*.npz"))) == 3

    main(argv) 
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == 16