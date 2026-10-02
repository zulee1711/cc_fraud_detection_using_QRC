class QRCProtocol:
    """
    Map temporal feature windows to measured quantum outputs.
    """

    def __init__(self, encoder, reservoir, observables, backend):
        if encoder.num_qubits != reservoir.num_input_qubits:
            raise ValueError("encoder and reservoir input sub-system must use the same number of qubits")
        if observables.num_qubits != reservoir.num_qubits:
            raise ValueError("observables must act on the same number of qubits as the reservoir (input + memory)")
        self.encoder = encoder
        self.reservoir = reservoir
        self.observables = observables
        self.backend = backend

    def run(self, windows):
        if windows.ndim != 3:
            raise ValueError("windows must have shape (samples, window_length, features)")
        return self.backend.run_batch(windows, self.encoder, self.reservoir, self.observables)