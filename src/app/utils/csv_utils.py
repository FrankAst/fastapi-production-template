"""CSV processing utilities for handling file uploads and data parsing."""

from io import StringIO

import pandas as pd
from fastapi import HTTPException, UploadFile, status
from pandas import DataFrame

from app.domain.constants import TARGET_COLUMN

# CSV validation constants
SUPPORTED_CSV_EXTENSION = ".csv"
MIN_REQUIRED_COLUMNS = 2
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_SIZE_MB = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)


def _validate_csv_filename_extension(filename: str | None) -> None:
    """Validate that the filename is present and ends with the .csv extension.

    Args:
        filename: The filename to validate.

    Raises:
        HTTPException: If the filename is None or doesn't end with .csv.
    """
    if not filename:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Filename is missing."
        )

    if not filename.lower().endswith(SUPPORTED_CSV_EXTENSION):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files (.csv) are supported.",
        )


def _raise_upload_too_large() -> None:
    """Raise HTTP 413 with the standard `MAX_UPLOAD_SIZE_MB` message.

    Raises:
        HTTPException: Always, with status 413.
    """
    raise HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB.",
    )


async def _read_and_decode_csv_content(file: UploadFile) -> str:
    """
    Read the uploaded file content, enforcing the upload size cap.

    Args:
        file: The uploaded file to read.

    Returns:
        str: The decoded CSV content as a string.

    Raises:
        HTTPException: 413 if the upload exceeds `MAX_UPLOAD_SIZE_BYTES`;
            400 for any other read/decode failure.
    """
    if file.size is not None and file.size > MAX_UPLOAD_SIZE_BYTES:
        _raise_upload_too_large()

    try:
        contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            _raise_upload_too_large()
        return contents.decode("utf-8")
    except HTTPException:
        raise
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
    Validate that the target column doesn't contain NaN values.

    Args:
        df: The DataFrame to validate.

    Raises:
        HTTPException: If the target column contains NaN values or is missing.
    """
    if TARGET_COLUMN not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required target column: '{TARGET_COLUMN}'.",
        )

    if df[TARGET_COLUMN].isna().any():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{TARGET_COLUMN}' contains missing values. "
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


def _validate_non_empty(df: DataFrame) -> None:
    """
    Reject CSVs with no data rows.

    Args:
        df: The DataFrame to validate.

    Raises:
        HTTPException: If the DataFrame has no rows.
    """
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no data rows.",
        )


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
