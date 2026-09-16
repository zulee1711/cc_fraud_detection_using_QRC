#%%
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import *

logger = get_logger(__name__)

#%%
def split_dataset(
    transactions: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15
    ):
    logger.info("Starting dataset splitting...")

    if train_ratio <= 0:
        raise ValueError(
            "train_ratio must be greater than 0."
        )

    if validation_ratio < 0:
        raise ValueError(
            "validation_ratio must be non-negative."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    if "TX_DATETIME" not in transactions.columns:
        raise ValueError(
            "Transaction dataset must contain 'TX_DATETIME'."
        )

    transactions = transactions.copy()

    transactions["TX_DATETIME"] = pd.to_datetime(
        transactions["TX_DATETIME"]
    )

    transactions = transactions.sort_values(
        "TX_DATETIME"
    ).reset_index(drop=True)

    first_date = (
        transactions["TX_DATETIME"]
        .dt.normalize()
        .min()
    )

    last_date = (
        transactions["TX_DATETIME"]
        .dt.normalize()
        .max()
    )

    total_days = (last_date - first_date).days + 1

    train_days = int(
        total_days * train_ratio
    )

    validation_days = int(
        total_days * validation_ratio
    )

    train_end = first_date + pd.Timedelta(
        days=train_days
    )

    validation_end = train_end + pd.Timedelta(
        days=validation_days
    )

    train = transactions[
        transactions["TX_DATETIME"] < train_end
        ].copy()

    validation = transactions[
        (transactions["TX_DATETIME"] >= train_end)
        & (transactions["TX_DATETIME"] < validation_end)
        ].copy()

    test = transactions[
        transactions["TX_DATETIME"] >= validation_end
        ].copy()

    logger.info(
        f"Dataset period: {first_date.date()} "
        f"to {last_date.date()} "
        f"({total_days} days)."
    )

    logger.info(
        f"Train period: {first_date.date()} "
        f"to {(train_end - pd.Timedelta(days=1)).date()} "
        f"({train_days} days)."
    )

    logger.info(
        f"Validation period: {train_end.date()} "
        f"to {(validation_end - pd.Timedelta(days=1)).date()} "
        f"({validation_days} days)."
    )

    logger.info(
        f"Test period: {validation_end.date()} "
        f"to {last_date.date()}."
    )

    logger.info(
        f"Transactions: "
        f"train={len(train):,}, "
        f"validation={len(validation):,}, "
        f"test={len(test):,}."
    )

    if "TX_FRAUD" in transactions.columns:
        logger.info(
            f"Fraud transactions: "
            f"train={train['TX_FRAUD'].sum():,}, "
            f"validation={validation['TX_FRAUD'].sum():,}, "
            f"test={test['TX_FRAUD'].sum():,}."
        )

        logger.info(
            f"Fraud rates: "
            f"train={train['TX_FRAUD'].mean():.4%}, "
            f"validation={validation['TX_FRAUD'].mean():.4%}, "
            f"test={test['TX_FRAUD'].mean():.4%}."
        )

    logger.info("Dataset split completed.")

    return train, validation, test

#%%
def main(
        transactions: pd.DataFrame,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        output_dir: Optional[str | Path] = None
):
    train, validation, test = split_dataset(
        transactions,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio
    )

    if output_dir is not None:
        output_dir = Path(output_dir)

        train_dir = output_dir / "train"
        validation_dir = output_dir / "validation"
        test_dir = output_dir / "test"

        logger.info(
            f"Saving dataset splits to {output_dir}"
        )

        dumping(
            train,
            output_dir / "transactions_train.pkl"
        )

        dumping(
            validation,
            output_dir / "transactions_validation.pkl"
        )

        dumping(
            test,
            output_dir / "transactions_test.pkl"
        )

        # Save training data by day
        for date, transactions_day in train.groupby(
                train["TX_DATETIME"].dt.date
        ):
            dumping(
                transactions_day.sort_values(
                    "TX_TIME_SECONDS"
                ),
                train_dir / f"{date}.pkl"
            )

        # Save validation data by day
        for date, transactions_day in validation.groupby(
                validation["TX_DATETIME"].dt.date
        ):
            dumping(
                transactions_day.sort_values(
                    "TX_TIME_SECONDS"
                ),
                validation_dir / f"{date}.pkl"
            )

        # Save test data by day
        for date, transactions_day in test.groupby(
                test["TX_DATETIME"].dt.date
        ):
            dumping(
                transactions_day.sort_values(
                    "TX_TIME_SECONDS"
                ),
                test_dir / f"{date}.pkl"
            )

        logger.info(
            "Dataset splitting pipeline completed."
        )

    return train, validation, test

#%%
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    raw_data_dir = Path.joinpath(project_root, "raw_data")
    processed_data_dir = Path.joinpath(project_root, "data")

    transactions = loading(
        raw_data_dir / "transactions.pkl"
    )

    train, validation, test = main(
        transactions,
        train_ratio=0.70,
        validation_ratio=0.15,
        output_dir=processed_data_dir
    )