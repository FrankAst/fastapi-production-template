"""Tests for CSV processing utilities.

This module tests the public CSV processing API through the process_csv_file function,
which provides the complete CSV processing pipeline: validation, reading, and parsing.
"""

from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile

from app.utils.csv_utils import process_csv_file

# Constants for test assertions
HTTP_400_BAD_REQUEST = 400
EXPECTED_ROW_COUNT = 2
EXPECTED_UNICODE_ROWS = 2
JOHN_AGE = 25
ALICE_SCORE = 95.5

"""Test cases for the complete CSV processing pipeline."""


@pytest.mark.anyio
async def test_process_csv_file_success_basic() -> None:
    """Test successful processing of a basic CSV file."""
    csv_data = "name,age\nJohn,25\nJane,30"

    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.csv"
    mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

    result = await process_csv_file(mock_file)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == EXPECTED_ROW_COUNT
    assert list(result.columns) == ["name", "age"]
    assert result.iloc[0]["name"] == "John"
    assert result.iloc[0]["age"] == JOHN_AGE
    assert result.iloc[1]["name"] == "Jane"
    mock_file.read.assert_called_once()


@pytest.mark.anyio
async def test_process_csv_file_success_complex() -> None:
    """Test successful processing of a more complex CSV file."""
    csv_data = "id,name,score,active\n1,Alice,95.5,true\n2,Bob,87.2,false"

    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "complex_data.csv"
    mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

    result = await process_csv_file(mock_file)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == EXPECTED_ROW_COUNT
    assert list(result.columns) == ["id", "name", "score", "active"]
    assert result.iloc[0]["name"] == "Alice"
    assert result.iloc[0]["score"] == ALICE_SCORE


@pytest.mark.anyio
async def test_process_csv_file_invalid_filename_txt() -> None:
    """Test that .txt files are rejected."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.txt"

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Only CSV files are supported" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_invalid_filename_xlsx() -> None:
    """Test that .xlsx files are rejected."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.xlsx"

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Only CSV files are supported" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_invalid_filename_json() -> None:
    """Test that .json files are rejected."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.json"

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Only CSV files are supported" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_none_filename() -> None:
    """Test that None filename is rejected."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = None

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Only CSV files are supported" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_empty_filename() -> None:
    """Test that empty filename is rejected."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = ""

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Only CSV files are supported" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_file_read_error() -> None:
    """Test handling of file read errors."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.csv"
    mock_file.read = AsyncMock(side_effect=Exception("File read error"))

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Error reading file content" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_decode_error() -> None:
    """Test handling of file decode errors."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.csv"
    mock_file.read = AsyncMock(return_value=b"\x80\x81\x82")  # Invalid UTF-8

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Error reading file content" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_empty_content() -> None:
    """Test handling of empty CSV content."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "empty.csv"
    mock_file.read = AsyncMock(return_value=b"")

    with pytest.raises(HTTPException) as exc_info:
        await process_csv_file(mock_file)

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST
    assert "Error parsing CSV" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_process_csv_file_malformed_csv_handled_gracefully() -> None:
    """Test that pandas handles malformed CSV gracefully."""
    # CSV with inconsistent columns - pandas is forgiving
    csv_data = "name,age\nJohn,25,extra_column\nJane"

    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "malformed.csv"
    mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

    # This should still work as pandas is quite forgiving
    result = await process_csv_file(mock_file)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.anyio
async def test_process_csv_file_with_headers_only() -> None:
    """Test CSV file with headers but no data rows."""
    csv_data = "name,age,city"

    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "headers_only.csv"
    mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

    result = await process_csv_file(mock_file)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0  # No data rows
    assert list(result.columns) == ["name", "age", "city"]
