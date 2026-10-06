"""PyTorch LSTM classifier for transaction sequences."""

from torch import nn


class LSTMClassifier(nn.Module):
    """Classify a sequence using the LSTM output at its final timestep."""

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values):
        sequence_outputs, _ = self.lstm(values)
        return self.output(sequence_outputs[:, -1, :]).squeeze(-1)