class QRCProtocol:
    """
    Map temporal feature windows to measured quantum outputs.
    """

    def __init__(self, encoder, reservoir, observables, backend):
        if encoder.num_qubits != reservoir.num_input_qubits:
            raise ValueError("encoder and reservoir input sub-system must use the same number of qubits")
        if any(obs.num_qubits != reservoir.num_qubits for obs in observables):
            raise ValueError("observables must act on the same number of qubits as the reservoir (input + memory)")
        self.encoder = encoder
        self.reservoir = reservoir
        self.observables = observables
        self.backend = backend

    def run(self, windows):
        return self.backend.run_batch(windows, self.encoder, self.reservoir, self.observables)