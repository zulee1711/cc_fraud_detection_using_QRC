#%%
import os
from pathlib import Path
import pandas as pd
from typing import Optional
from ..logger import get_logger


def coerce_numeric_columns(df, numeric_cols: list[str]):
    """Convert numeric-looking object columns to numeric safely."""
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

logger = get_logger(__name__)

#%%
def loading(
        filename: str | Path,
        file_ext: Optional[str] = None,
        **kwargs
):
    """
    Load a file depending on its file extension.

    Supported formats:
        - csv
        - xlsx
        - parquet
        - pkl / pickle

    Parameters
    ----------
    filename : str | Path
        Path to the file.

    file_ext : str, optional
        File extension. If not provided, it is inferred
        from filename.

    Returns
    -------
    pandas.DataFrame
    """
    filename = Path(filename)

    if file_ext is None:
        file_ext = filename.suffix.lower().lstrip(".")

    if file_ext == "csv":
        data = pd.read_csv(
            filename,
            low_memory=False,
            **kwargs
        )

    elif file_ext == "xlsx":
        data = pd.read_excel(
            filename,
            **kwargs
        )

    elif file_ext in ("parquet", "prq"):
        data = pd.read_parquet(
            filename,
            **kwargs
        )

    elif file_ext in ("pkl", "pickle"):
        data = pd.read_pickle(
            filename,
            **kwargs
        )

    else:
        logger.error(
            f"{file_ext} is not supported."
        )
        return None

    logger.debug(f"Loaded file: {filename}")

    return data

#%%
def load_daily_files(
        input_dir:str | Path,
        start_date:Optional[str] = None,
        end_date:Optional[str] = None,
):
    input_dir = Path(input_dir)

    files = sorted(input_dir.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}.")

    start_date = str(start_date) if start_date else files[0].stem
    end_date = str(end_date) if end_date else files[-1].stem

    logger.info(f"Loading files from {start_date} to {end_date}")
    files = [f for f in files if str(start_date) <= f.stem <= str(end_date)]

    if not files:
        raise FileNotFoundError(
            f"No .pkl files found in {input_dir} "
            f"between {start_date} and {end_date}."
        )

    df = pd.concat(
        [loading(f) for f in files],
        ignore_index=True,
    )

    NUMERIC_COLUMNS = [
        'TRANSACTION_ID',
        'TX_AMOUNT',
        'TX_TIME_SECONDS',
        'TX_TIME_DAYS',
        'TX_FRAUD',
        'TX_FRAUD_SCENARIO',
    ]

    df = coerce_numeric_columns(df, NUMERIC_COLUMNS)
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

    df_final = (
        df
        .sort_values("TX_DATETIME")
        .reset_index(drop=True)
    )

    # -1 represents missing values in the real-world data.
    # Replace with 0 to match the original simulator code.
    # df_final = df_final.replace(-1, 0)

    logger.info(
        f"Loaded {len(df_final):,} transactions."
    )

    return df_final

#%%
def load_splits(data_root):
    root = Path(data_root)
    return {
        s: load_daily_files(root / s)
        for s in ('train', 'validation', 'test')
    }

#%%
def dumping(
        data,
        out_path: str | Path,
        file_ext: str = None,
        **kwargs
):
    """
    Save data to a file.

    Supported formats:
        - csv
        - xlsx
        - parquet
        - pkl / pickle

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, dict, or str
        Data to save.

    out_path : str | Path
        Output file path.

    file_ext : str, optional
        File extension. If not provided, it is
        inferred from out_path.
    """

    out_path = Path(out_path)

    # Infer extension from output path
    if file_ext is None:
        file_ext = out_path.suffix.lower().lstrip(".")

    # Create output directory if necessary
    out_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # Convert dictionaries to DataFrames
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    # Save text
    if isinstance(data, str):

        with open(out_path, "w") as f:
            f.write(data)

        logger.info(
            f"Output saved to {out_path}"
        )

        return

    # Save pandas objects
    if isinstance(data, (pd.DataFrame, pd.Series)):

        if file_ext == "csv":

            data.to_csv(
                out_path,
                index=False,
                **kwargs
            )

        elif file_ext == "xlsx":

            data.to_excel(
                out_path,
                index=False,
                **kwargs
            )

        elif file_ext in ("parquet", "prq"):

            data.to_parquet(
                out_path,
                index=False,
                **kwargs
            )

        elif file_ext in ("pkl", "pickle"):

            data.to_pickle(
                out_path,
                **kwargs
            )

        else:

            logger.error(
                f"{file_ext} is not supported."
            )

            return

        logger.info(
            f"Output saved to {out_path}"
        )

        return

    logger.error(
        f"Cannot recognize data type: "
        f"{type(data)}"
    )