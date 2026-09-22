class QRCProtocol:
    """
    Map temporal feature windows to measured quantum outputs.
    """

    def __init__(self, encoder, reservoir, backend):
        if encoder.num_qubits != reservoir.num_input_qubits:
            raise ValueError("encoder and reservoir input sub-system must use the same number of qubits")
        self.encoder = encoder
        self.reservoir = reservoir
        self.backend = backend

    def run(self, windows):
        return self.backend.run_batch(windows, self.encoder, self.reservoir)