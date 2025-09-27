"""CSV processing utilities for handling file uploads and data parsing."""

from io import StringIO

import pandas as pd
from fastapi import HTTPException, UploadFile
from pandas import DataFrame


def _validate_csv_filename(filename: str | None) -> None:
    """
    Validate that the uploaded file has a .csv extension.

    Args:
        filename: The filename to validate.

    Raises:
        HTTPException: If the filename is None or doesn't end with .csv.
    """
    if not filename or not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")


async def _read_csv_content(file: UploadFile) -> str:
    """
    Read and decode the content of an uploaded CSV file.

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


async def process_csv_file(file: UploadFile) -> DataFrame:
    """
    Complete CSV processing pipeline: validate filename, read content, and parse
    to DataFrame.

    Args:
        file: The uploaded CSV file to process.

    Returns:
        DataFrame: The processed CSV data as a pandas DataFrame.
    """
    _validate_csv_filename(file.filename)
    csv_content = await _read_csv_content(file)
    return _parse_csv_to_dataframe(csv_content)
