import numpy as np
import pandas as pd

from qrc.processing import DataProcessor


def _frame(values, labels, customers, timestamps):
    return pd.DataFrame(
        {
            "feature": values,
            "TX_FRAUD": labels,
            "CUSTOMER_ID": customers,
            "TX_DATETIME": pd.to_datetime(timestamps),
        }
    )


def test_processor_preserves_timestamps_as_row_aligned_metadata():
    train = _frame(
        [1.0, 2.0],
        [0, 1],
        [10, 11],
        ["2026-01-01 09:00", "2026-01-01 10:00"],
    )
    validation = _frame(
        [1.5],
        [0],
        [12],
        ["2026-01-02 09:00"],
    )
    test = _frame(
        [2.5],
        [1],
        [13],
        ["2026-01-03 09:00"],
    )

    result = DataProcessor({1: ["feature"]}).process(
        train,
        validation,
        test,
        feature_set=1,
    )

    np.testing.assert_array_equal(
        result["timestamps_train"], train["TX_DATETIME"].to_numpy()
    )
    np.testing.assert_array_equal(
        result["timestamps_validation"], validation["TX_DATETIME"].to_numpy()
    )
    np.testing.assert_array_equal(
        result["timestamps_test"], test["TX_DATETIME"].to_numpy()
    )

    # Timestamps remain metadata and are not added to the model features.
    assert result["X_train"].shape == (2, 1)
    assert result["features"] == ["feature"]

def test_processor_preserves_transaction_ids_when_present():
    train = _frame(
        [1.0, 2.0],
        [0, 1],
        [10, 11],
        ["2026-01-01 09:00", "2026-01-01 10:00"],
    )
    validation = _frame(
        [1.5],
        [0],
        [12],
        ["2026-01-02 09:00"],
    )
    test = _frame(
        [2.5],
        [1],
        [13],
        ["2026-01-03 09:00"],
    )

    train["TRANSACTION_ID"] = [100, 101]
    validation["TRANSACTION_ID"] = [102]
    test["TRANSACTION_ID"] = [103]

    result = DataProcessor({1: ["feature"]}).process(
        train,
        validation,
        test,
        feature_set=1,
    )

    np.testing.assert_array_equal(
        result["ids_train"],
        np.array([100, 101]),
    )
    np.testing.assert_array_equal(
        result["ids_validation"],
        np.array([102]),
    )
    np.testing.assert_array_equal(
        result["ids_test"],
        np.array([103]),
    )