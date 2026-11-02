import numpy as np
import pytest

from qrc.protocol import QRCProtocol


@pytest.fixture
def stub_reservoir(monkeypatch):
    """Random features of the right shape instead of simulating the reservoir.

    For tests of the bookkeeping around a QRC run (studies, summaries, run
    directories), where simulating ~14k windows is most of the time and none of
    the assertions.
    """
    def run(self, windows):
        rng = np.random.default_rng(len(windows))
        return rng.normal(size=(len(windows), len(self.observables)))

    monkeypatch.setattr(QRCProtocol, "run", run)
