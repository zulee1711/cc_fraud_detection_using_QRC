import math
import numpy as np
from qiskit import QuantumCircuit


class Encoding:

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits


class AngleEncoding(Encoding):

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

        super().__init__(num_qubits)
        self.axis = axis
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
            raise ValueError(
                f"Input length {len(input)} does not match number of qubits {self.num_qubits}."
            )

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


class ReuploadingEncoding(Encoding):

    def __init__(
        self,
        num_qubits,
        num_layers=2,
        axis="y",
        scaling=np.pi,
        interleave_axis="z",
        rng=None,
    ):
        """Data re-uploading: the same input is angle-encoded `num_layers` times, with fixed
        random rotations about `interleave_axis` in between. Each feature then enters the
        state with frequencies up to `num_layers * scaling` instead of only `scaling`.
        `num_layers=1` is exactly `AngleEncoding`.

        Args:
            num_qubits (int): number of qubits or features.
            num_layers (int): how many times the input is encoded.
            axis (str): rotation axis of each encoding layer, see `AngleEncoding`.
            scaling (float): scaling factor of each encoding layer, see `AngleEncoding`.
            interleave_axis (str): axis of the fixed rotations between layers; must differ
                from `axis`, otherwise the rotations commute and the layers just add up.
            rng: seed or numpy Generator fixing the interleaved angles.
        """
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if interleave_axis == axis:
            raise ValueError(
                f"interleave_axis must differ from axis, both are '{axis}'"
            )

        super().__init__(num_qubits)
        self.layers = [
            AngleEncoding(num_qubits, axis=axis, scaling=scaling)
            for _ in range(num_layers)
        ]
        self.interleave_axis = interleave_axis
        rng = np.random.default_rng(rng)
        self.interleave_angles = rng.uniform(0, 2 * np.pi, (num_layers - 1, num_qubits))

    def set_scaling_based_on_input_range(self, min_value, max_value):
        """Sets the scaling factor of every layer, see `AngleEncoding`."""
        for layer in self.layers:
            layer.set_scaling_based_on_input_range(min_value, max_value)

    def encode(self, input, min_value=0.0):
        """Creates the re-uploading circuit for the given input data.

        Args:
            input (list): list of feature data, length must match num_qubits.
            min_value (float): minimum value of the input range, see `AngleEncoding`.

        Returns:
            encoding circuit to be prepended to qrc
        """
        qc = QuantumCircuit(self.num_qubits)
        for l, layer in enumerate(self.layers):
            layer_qc = layer.encode(input, min_value)
            if layer_qc is None:
                return
            qc.compose(layer_qc, inplace=True)
            if l < len(self.layers) - 1:
                for i, angle in enumerate(self.interleave_angles[l]):
                    if self.interleave_axis == "x":
                        qc.rx(angle, i)
                    elif self.interleave_axis == "y":
                        qc.ry(angle, i)
                    elif self.interleave_axis == "z":
                        qc.rz(angle, i)

        return qc


class DenseAngleEncoding(Encoding):
    """Dense angle encoding (DAE) is a combination of angle encoding and phase encoding.
    DAE allows two feature values to be encoded in a single qubit.
    """

    def __init__(self, num_qubits, axis1="y", axis2="z", scaling=np.pi):
        """
        Args:
            num_qubits (int): number of qubits or features.
            axis1 (str): rotation axis for the first feature on one qubit, e.g. 'x', 'y', or 'z'.
            axis2 (str): rotation axis for the second feature on the same qubit, e.g. 'x', 'y', or 'z'.
            scaling (float): scaling factor applied to input, see `AngleEncoding`.
        """

        if axis1 == axis2:
            raise ValueError(f"axis1 and axis2 must differ, both are '{axis1}'")

        super().__init__(num_qubits)
        self.axis1 = axis1
        self.axis2 = axis2
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
        """Creates a dense angle encoding circuit for the given input data.

        Args:
            input (list): list of feature data, number of qubits < len(input) <= 2 * num_qubits.
            min_value (float): minimum value of the input range (optional, only used if input
            is not between 0 and 1).

        Returns:
            encoding circuit to be prepended to qrc
        """

        if len(input) <= self.num_qubits:
            raise ValueError(
                f"Input length {len(input)} must be greater than number of qubits {self.num_qubits}."
            )
        if len(input) > 2 * self.num_qubits:
            raise ValueError(
                f"Input length {len(input)} must be at most {2 * self.num_qubits}."
            )

        qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            value1 = input[i]
            angle1 = (value1 - min_value) * self.scaling
            if self.axis1 == "x":
                qc.rx(angle1, i)
            elif self.axis1 == "y":
                qc.ry(angle1, i)
            elif self.axis1 == "z":
                qc.rz(angle1, i)

            if self.num_qubits + i < len(input):
                value2 = input[self.num_qubits + i]
                angle2 = (value2 - min_value) * self.scaling
                if self.axis2 == "x":
                    qc.rx(angle2, i)
                elif self.axis2 == "y":
                    qc.ry(angle2, i)
                elif self.axis2 == "z":
                    qc.rz(angle2, i)

        return qc
