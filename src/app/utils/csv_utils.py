"""CSV processing utilities for handling file uploads and data parsing."""

from io import StringIO

import pandas as pd
from fastapi import UploadFile
from pandas import DataFrame

from app.domain.constants import TARGET_COLUMN

from .csv_exceptions import CsvContentError, CsvFormatError, CsvSizeError

SUPPORTED_CSV_EXTENSION = ".csv"
MIN_REQUIRED_COLUMNS = 2
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_SIZE_MB = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)


def _validate_csv_filename_extension(filename: str | None) -> None:
    """Validate that the filename is present and ends with the .csv extension.

    Args:
        filename: The filename to validate.

    Raises:
        CsvFormatError: If the filename is None or doesn't end with .csv.
    """
    if not filename:
        msg = "Invalid file type. Filename is missing."
        raise CsvFormatError(msg)

    if not filename.lower().endswith(SUPPORTED_CSV_EXTENSION):
        msg = "Invalid file type. Only CSV files (.csv) are supported."
        raise CsvFormatError(msg)


def _raise_upload_too_large() -> None:
    """Raise CsvSizeError with the standard `MAX_UPLOAD_SIZE_MB` message.

    Raises:
        CsvSizeError: Always.
    """
    msg = f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB."
    raise CsvSizeError(msg)


async def _read_and_decode_csv_content(file: UploadFile) -> str:
    """
    Read the uploaded file content, enforcing the upload size cap.

    Args:
        file: The uploaded file to read.

    Returns:
        str: The decoded CSV content as a string.

    Raises:
        CsvSizeError: If the upload exceeds `MAX_UPLOAD_SIZE_BYTES`.
        CsvFormatError: For any other read/decode failure.
    """
    if file.size is not None and file.size > MAX_UPLOAD_SIZE_BYTES:
        _raise_upload_too_large()

    try:
        contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            _raise_upload_too_large()
        return contents.decode("utf-8")
    except CsvSizeError:
        raise
    except Exception as err:
        msg = f"Error reading file content: {err!s}"
        raise CsvFormatError(msg) from err


def _parse_csv_to_dataframe(csv_content: str) -> DataFrame:
    """
    Parse CSV string content into a pandas DataFrame.

    Args:
        csv_content: The CSV content as a string.

    Returns:
        DataFrame: The parsed CSV data as a pandas DataFrame.

    Raises:
        CsvFormatError: If there's an error parsing the CSV content.
    """
    try:
        return pd.read_csv(StringIO(csv_content))
    except Exception as err:
        msg = f"Error parsing CSV: {err!s}"
        raise CsvFormatError(msg) from err


def _validate_target_column(df: DataFrame) -> None:
    """
    Validate that the target column doesn't contain NaN values.

    Args:
        df: The DataFrame to validate.

    Raises:
        CsvContentError: If the target column contains NaN values or is missing.
    """
    if TARGET_COLUMN not in df.columns:
        msg = f"Missing required target column: '{TARGET_COLUMN}'."
        raise CsvContentError(msg)

    if df[TARGET_COLUMN].isna().any():
        msg = (
            f"Target column '{TARGET_COLUMN}' contains missing values. "
            "Please ensure all target values are provided."
        )
        raise CsvContentError(msg)


def _validate_number_of_columns(df: DataFrame) -> None:
    """
    Validate that the DataFrame has at least two columns.

    Args:
        df: The DataFrame to validate.

    Raises:
        CsvContentError: If the DataFrame has fewer than two columns.
    """
    if df.shape[1] < MIN_REQUIRED_COLUMNS:
        msg = "CSV must contain at least two columns (features and target)."
        raise CsvContentError(msg)


def _validate_non_empty(df: DataFrame) -> None:
    """
    Reject CSVs with no data rows.

    Args:
        df: The DataFrame to validate.

    Raises:
        CsvContentError: If the DataFrame has no rows.
    """
    if df.empty:
        msg = "CSV contains no data rows."
        raise CsvContentError(msg)


async def process_csv_file(
    file: UploadFile, *, require_target: bool = True
) -> DataFrame:
    """
    Complete CSV processing pipeline: validate filename, read content, parse
    to DataFrame, and optionally validate the target column.

    Args:
        file: The uploaded CSV file to process.
        require_target: When True (default), enforce that the target column is
            present and free of NaN — required for /train. Set to False for
            inference paths like /batch where labels are not provided.

    Returns:
        DataFrame: The processed and validated CSV data as a pandas DataFrame.
    """
    _validate_csv_filename_extension(file.filename)
    csv_content = await _read_and_decode_csv_content(file)
    df = _parse_csv_to_dataframe(csv_content)
    _validate_number_of_columns(df)
    _validate_non_empty(df)
    if require_target:
        _validate_target_column(df)

    return df
