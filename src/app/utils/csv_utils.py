"""CSV processing utilities for handling file uploads and data parsing."""

from io import StringIO

import pandas as pd
from fastapi import HTTPException, UploadFile
from pandas import DataFrame

# CSV validation constants
SUPPORTED_CSV_EXTENSION = ".csv"
MIN_REQUIRED_COLUMNS = 2
"""Minimum number of columns required in training CSV files.

The CSV must have at least two columns: one or more feature columns
and one target column.
"""


def _validate_csv_filename_extension(filename: str | None) -> None:
    """

    Args:
        filename: The filename to validate.

    Raises:
        HTTPException: If the filename is None or doesn't end with .csv.
    """
    if not filename:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Filename is missing."
        )

    if not filename.endswith(SUPPORTED_CSV_EXTENSION):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files (.csv) are supported.",
        )


async def _read_and_decode_csv_content(file: UploadFile) -> str:
    """

    Args:
        file: The uploaded file to read.

    Returns:
        str: The decoded CSV content as a string.

    Raises:
        HTTPException: If there's an error reading or decoding the file.
    """
    try:
        contents = await file.read()
        return contents.decode("utf-8")
    except Exception as err:
        raise HTTPException(
            status_code=400, detail=f"Error reading file content: {err!s}"
        ) from err


def _parse_csv_to_dataframe(csv_content: str) -> DataFrame:
    """
    Parse CSV string content into a pandas DataFrame.

    Args:
        csv_content: The CSV content as a string.

    Returns:
        DataFrame: The parsed CSV data as a pandas DataFrame.

    Raises:
        HTTPException: If there's an error parsing the CSV content.
    """
    try:
        return pd.read_csv(StringIO(csv_content))
    except Exception as err:
        raise HTTPException(
            status_code=400, detail=f"Error parsing CSV: {err!s}"
        ) from err


def _validate_target_column(df: DataFrame) -> None:
    """
    Validate that the target column (last column) doesn't contain NaN values.

    Args:
        df: The DataFrame to validate.

    Raises:
        HTTPException: If the target column contains NaN values.
    """
    if df.empty:
        return

    target_column = df.iloc[:, -1]
    if target_column.isna().any():
        raise HTTPException(
            status_code=400,
            detail=(
                "Target column (last column) contains missing values. "
                "Please ensure all target values are provided."
            ),
        )


def _validate_number_of_columns(df: DataFrame) -> None:
    """
    Validate that the DataFrame has at least two columns.

    Args:
        df: The DataFrame to validate.

    Raises:
        HTTPException: If the DataFrame has fewer than two columns.
    """
    if df.shape[1] < MIN_REQUIRED_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain at least two columns (features and target).",
        )


async def process_csv_file(file: UploadFile) -> DataFrame:
    """
    Complete CSV processing pipeline: validate filename, read content, parse
    to DataFrame, and validate target column.

    Args:
        file: The uploaded CSV file to process.

    Returns:
        DataFrame: The processed and validated CSV data as a pandas DataFrame.
    """
    _validate_csv_filename_extension(file.filename)
    csv_content = await _read_and_decode_csv_content(file)
    df = _parse_csv_to_dataframe(csv_content)
    _validate_target_column(df)
    _validate_number_of_columns(df)

    return df
