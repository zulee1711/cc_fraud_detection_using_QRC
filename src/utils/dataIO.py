#%%
import os
from pathlib import Path
import pandas as pd
from typing import Optional
from .logger import get_logger

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
        start_date:str,
        end_date:str,
):
    input_dir = Path(input_dir)

    begin_file = f"{start_date}.pkl"
    end_file = f"{end_date}.pkl"

    files = sorted(
        file for file in input_dir.iterdir()
        if file.suffix == ".pkl"
        and begin_file <= file.name <= end_file
    )

    if not files:
        raise FileNotFoundError(
            f"No .pkl files found in {input_dir} "
            f"between {start_date} and {end_date}."
        )

    logger.info(
        f"Reading {len(files)} daily files from "
        f"{start_date} to {end_date}."
    )

    frames = []

    for file in files:
        logger.info(f"Loading {file.name}")

        df = loading(file)
        frames.append(df)

    df_final = pd.concat(
        frames,
        ignore_index=True
    )

    df_final = (
        df_final
        .sort_values("TRANSACTION_ID")
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