import math
import numpy as np
from qiskit import QuantumCircuit


class AngleEncoding:

    def __init__(self, num_qubits, axis="y", scaling=np.pi):
        """
        Args:
            num_qubits (int): number of qubits or features.
            axis (str): rotation axis, e.g. 'x', 'y', or 'z'.
            scaling (float): scaling factor applied to raw input,
                important: default assumes all real input is a float between 0.0-1.0, thus
                we get angles in the range [0, π] for the rotation. If input is in a
                different range, adjust this scaling factor accordingly.
        """

        self.axis = axis
        self.num_qubits = num_qubits
        self.scaling = scaling

    def set_scaling_based_on_input_range(self, min_value, max_value):
        """Sets the scaling factor based on the input range, in the case we don't have input
        between 0 and 1.

        Args:
            min_value (float): minimum value of the input range.
            max_value (float): maximum value of the input range.
        """

        self.scaling = math.pi / (max_value - min_value)

    def encode(self, input, min_value=0.0):
        """Creates an angle encoding circuit for the given input data.

        Args:
            input (list): list of feature data, length must match num_qubits.
            min_value (float): minimum value of the input range (optional, only used if input
            is not between 0 and 1).

        Returns:
            encoding circuit to be prepended to qrc
        """
        if len(input) != self.num_qubits:
            print(
                f"Input length {len(input)} does not match number of qubits {self.num_qubits}."
            )
            return

        qc = QuantumCircuit(self.num_qubits)
        for i, value in enumerate(input):
            angle = (value - min_value) * self.scaling
            if self.axis == "x":
                qc.rx(angle, i)
            elif self.axis == "y":
                qc.ry(angle, i)
            elif self.axis == "z":
                qc.rz(angle, i)

        return qc
