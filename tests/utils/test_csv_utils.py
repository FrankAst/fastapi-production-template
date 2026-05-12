"""Tests for CSV processing utilities.

This module tests the public CSV processing API through the process_csv_file function,
which provides the complete CSV processing pipeline: validation, reading, and parsing.
"""

from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile, status

from app.utils.csv_utils import process_csv_file
from tests.utils.conftest import CsvErrorTestData, CsvTestData

# Constants for test assertions
EXPECTED_ROW_COUNT = 2

"""Test cases for the complete CSV processing pipeline."""


# Valid CSV processing tests ###
@pytest.mark.anyio
async def test_process_csv_file_success(mock_valid_csv_data: CsvTestData) -> None:
    """
    Test successful processing of various CSV file formats.
    """
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = mock_valid_csv_data.filename
    mock_file.size = None
    mock_file.read = AsyncMock(
        return_value=mock_valid_csv_data.csv_data.encode("utf-8")
    )

    result = await process_csv_file(mock_file)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == EXPECTED_ROW_COUNT
    assert list(result.columns) == mock_valid_csv_data.expected_columns
    mock_file.read.assert_called_once()


@pytest.mark.anyio
async def test_process_csv_file_invalid_filename(mock_invalid_file: Mock) -> None:
    """Test handling of invalid filenames."""
    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_invalid_file)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid file type" in str(exc_info.value.detail)
    mock_invalid_file.read.assert_not_called()


# Error handling tests ###


@pytest.mark.anyio
async def test_process_csv_file_error_scenarios(
    mock_error_csv_scenarios: CsvErrorTestData,
) -> None:
    """Test various CSV error scenarios using parametrized fixture."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = mock_error_csv_scenarios.filename
    mock_file.size = None
    mock_file.read = AsyncMock(return_value=mock_error_csv_scenarios.read_value)

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert mock_error_csv_scenarios.expected_message in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_insufficient_columns() -> None:
    """Test CSV with only one column is rejected."""
    csv_data = "single_column\nvalue1\nvalue2"  # Only 1 column

    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "single_column.csv"
    mock_file.size = None
    mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "CSV must contain at least two columns" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_rejects_headers_only() -> None:
    """Test CSV file with headers but no data rows is rejected."""
    csv_data = "name,age,city"

    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "headers_only.csv"
    mock_file.size = None
    mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "CSV contains no data rows" in str(exc_info.value.detail)


@pytest.mark.parametrize(
    "size_bytes",
    [
        pytest.param(50 * 1024 * 1024 + 1, id="just_over_limit"),
        pytest.param(200 * 1024 * 1024, id="way_over_limit"),
    ],
)
@pytest.mark.anyio
async def test_process_csv_file_rejects_oversize(size_bytes: int) -> None:
    """Test CSV files exceeding the 50 MB cap are rejected with HTTP 413."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "huge.csv"
    mock_file.size = size_bytes

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "File too large" in str(exc_info.value.detail)
