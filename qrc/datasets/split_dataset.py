#%%
from pathlib import Path
from typing import Optional

import pandas as pd

from ..logger import get_logger
from .dataIO import dumping

logger = get_logger(__name__)

#%%
def split_dataset(
    transactions: pd.DataFrame,
    validation_start: Optional[str | pd.Timestamp] = None,
    test_start: Optional[str | pd.Timestamp] = None
    ):
    """Transactions split by specified dates:
    -train: from first transaction until last transaction before validation_start 
    -validation: from validation_start until last transaction before test_start
    -test: from test_start until last transaction
    
    Note: split will only work with datasets where all transactions are on same year. If more than one """
    logger.info("Starting dataset splitting...")

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
    
    start_month = first_date.replace(day=1)

    if last_date >= first_date + pd.DateOffset(years=1):
        raise ValueError(
            f"Transactions must cover at most one year, but they span "
            f"from {first_date.date()} to {last_date.date()}."
        )
    

    train_end = (
        start_month + pd.DateOffset(months=7)
        if validation_start is None
        else pd.Timestamp(validation_start).normalize()
    )

    validation_end = (
        start_month + pd.DateOffset(months=8)
        if test_start is None
        else pd.Timestamp(test_start).normalize()
    )
    if not first_date < train_end < validation_end <= last_date:
        raise ValueError(
            f"Split dates must satisfy first transaction date "
            f"({first_date.date()}) < validation_start "
            f"({train_end.date()}) < test_start "
            f"({validation_end.date()}) <= last transaction date "
            f"({last_date.date()})."
        )
    
    total_days = (last_date - first_date).days + 1
    train_days = (train_end - first_date).days
    validation_days = (validation_end - train_end).days

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
        validation_start: Optional[str | pd.Timestamp] = None,
        test_start: Optional[str | pd.Timestamp] = None,
        output_dir: Optional[str | Path] = None
):
    train, validation, test = split_dataset(
        transactions,
        validation_start=validation_start,
        test_start=test_start
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
def split_features_by_dates(
    full_features: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    datetime_column: str = "TX_DATETIME",
):
    full_features = full_features.copy()

    full_features[datetime_column] = pd.to_datetime(
        full_features[datetime_column]
    )

    train_dt = pd.to_datetime(train[datetime_column])
    validation_dt = pd.to_datetime(validation[datetime_column])
    test_dt = pd.to_datetime(test[datetime_column])

    train_start = train_dt.dt.normalize().min()
    validation_start = validation_dt.dt.normalize().min()
    test_start = test_dt.dt.normalize().min()

    train_features = full_features[
        (full_features[datetime_column] >= train_start)
        & (full_features[datetime_column] < validation_start)
        ].copy()

    validation_features = full_features[
        (full_features[datetime_column] >= validation_start)
        & (full_features[datetime_column] < test_start)
        ].copy()

    test_features = full_features[
        full_features[datetime_column] >= test_start
        ].copy()

    train_features = (
        train_features
        .sort_values(datetime_column)
        .reset_index(drop=True)
    )

    validation_features = (
        validation_features
        .sort_values(datetime_column)
        .reset_index(drop=True)
    )

    test_features = (
        test_features
        .sort_values(datetime_column)
        .reset_index(drop=True)
    )

    return (
        train_features,
        validation_features,
        test_features,
    )

# #%%
# if __name__ == "__main__":
#     project_root = Path(__file__).resolve().parents[2]
#
#     raw_data_dir = Path.joinpath(project_root, "raw_data")
#     processed_data_dir = Path.joinpath(project_root, "data")
#
#     transactions = loading(
#         raw_data_dir / "transactions.pkl"
#     )
#
#     train, validation, test = main(
#         transactions,
#         train_ratio=0.70,
#         validation_ratio=0.15,
#         output_dir=processed_data_dir
#     )


