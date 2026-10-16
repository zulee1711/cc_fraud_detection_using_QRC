import numpy as np
import pytest

from qrc.sequences import window_by_customer_id


def _input_data():
    return {
        "X_train": np.array([[1.0], [2.0]]),
        "X_validation": np.array([[3.0]]),
        "X_test": np.array([[4.0]]),
        "y_train": np.array([0, 1]),
        "y_validation": np.array([0]),
        "y_test": np.array([1]),
        "groups_train": np.array([7, 7]),
        "groups_validation": np.array([7]),
        "groups_test": np.array([7]),
        "timestamps_train": np.array(
            ["2026-01-01", "2026-01-02"], dtype="datetime64[D]"
        ),
        "timestamps_validation": np.array(["2026-01-03"], dtype="datetime64[D]"),
        "timestamps_test": np.array(["2026-01-04"], dtype="datetime64[D]"),
        "features": ["feature"],
    }


def test_windowing_preserves_timestamp_alignment():
    data = _input_data()

    result = window_by_customer_id(data, length=2)

    for split in ("train", "validation", "test"):
        np.testing.assert_array_equal(
            result[f"timestamps_{split}"],
            data[f"timestamps_{split}"],
        )
        assert len(result[f"timestamps_{split}"]) == len(result[f"y_{split}"])


def test_windowing_rejects_misaligned_timestamps():
    data = _input_data()
    data["timestamps_validation"] = np.array(
        ["2026-01-03", "2026-01-04"], dtype="datetime64[D]"
    )

    with pytest.raises(ValueError, match="timestamps"):
        window_by_customer_id(data, length=2)
