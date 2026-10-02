import numpy as np
from qiskit.quantum_info import PauliList
from qrc.encodings import Encoding
from qrc.reservoirs import Reservoir

from abc import ABC, abstractmethod

class Backend(ABC):

    @abstractmethod
    def run_batch(self, windows: np.ndarray, encoder: Encoding, reservoir: Reservoir, observables: PauliList) -> np.ndarray:
        """
        Runs the circuits and returns the (estimated) expectation values for a batch of transaction windows.
        Each transaction window corresponds to an array of L transactions, each of them being described
        by an array of features.

        Args:
            windows (np.ndarray): batch of transaction windows. Has shape ``(samples, window_length, features)``.
            encoder (Encoding): input encoding.
            reservoir (Reservoir): reservoir used for the QRC pipeline.
            observables (PauliList): Pauli observables for which we want the (estimated) expectation value.

        Returns:
            (np.ndarray): (Estimated) expectation values for the requested observables.
        """
