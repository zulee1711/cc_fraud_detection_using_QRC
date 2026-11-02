"""E6 noise-model script (experiments/spinpulse_noise_models.py)."""

import csv

import numpy as np
import pytest
from qiskit.quantum_info import PauliList

pytest.importorskip("spin_pulse")

from experiments.spinpulse_noise_models import depolarizing_features, depolarizing_model, main
from qrc.backends.numpy_backend import NumpyBackend
from qrc.encodings import AngleEncoding
from qrc.hamiltonians import tfim
from qrc.protocol import QRCProtocol
from qrc.reservoirs import HamiltonianReservoir


def test_depolarizing_model_without_noise_is_the_ideal_circuit():
    """The bridge to the P2 simulator: at fidelity 1 it must reproduce NumpyBackend exactly
    (same reservoir, encoding, reset and qubit order)."""
    reservoir = HamiltonianReservoir(tfim(3, rng=1), 1, time=1.0, steps=2)
    observables = PauliList(["IIZ", "IZI", "ZII", "IZZ", "ZZI", "IIX"])
    X = np.random.default_rng(0).random((6, 4, 1))
    expected = QRCProtocol(AngleEncoding(1), reservoir, observables, NumpyBackend(reset_inputs=True)).run(X)
    got = depolarizing_features(depolarizing_model(), X, reservoir, observables, 1.0, 0.0, None)
    np.testing.assert_allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("noise", ["quasistatic", "white"])
def test_calibration_finds_a_t2_close_to_the_target(tmp_path, noise):
    out = tmp_path / "calibration.csv"
    main(["calibrate", "--noise-types", noise, "--targets", "0.99", "--samples", "20",
          "--iterations", "6", "--calibration", str(out)])
    with open(out, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert float(row["f2"]) == pytest.approx(0.99, abs=0.005)  # loose: 6 steps, 20 samples
    assert float(row["f1"]) > float(row["f2"])  # 1q pulses are shorter, so less dephased


def test_compare_writes_one_row_per_model_precision_and_delay(tmp_path):
    calibration = tmp_path / "calibration.csv"
    calibration.write_text("noise,target_f2,t2_us,f2,f1,samples\n"
                           "pink,0.98,5.0,0.98,0.999,1\nquasistatic,0.98,5.0,0.98,0.999,1\n")
    out = tmp_path / "compare.csv"
    main(["compare", "--noise-types", "pink", "quasistatic", "--targets", "0.98", "--seeds", "42",
          "--precisions", "0", "0.05", "--noise-samples", "2", "--n-mem-qubits", "1",
          "--window-length", "3", "--length", "60", "--calibration", str(calibration), "--out", str(out)])
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    # 4 models (ideal, depolarizing, sp-pink, sp-quasistatic) x 2 precisions x 3 delays
    assert len(rows) == 24
    assert {r["model"] for r in rows} == {"ideal", "depolarizing", "sp-pink", "sp-quasistatic"}
    assert all(0.0 <= float(r["capacity"]) <= 1.0 for r in rows)