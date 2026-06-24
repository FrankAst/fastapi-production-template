"""Tests for CSV processing utilities.

This module tests the public CSV processing API through the process_csv_file function,
which provides the complete CSV processing pipeline: validation, reading, and parsing.
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from app.utils import CsvContentError, CsvFormatError, CsvSizeError
from app.utils.csv_utils import process_csv_file
from tests.utils.conftest import (
    CsvErrorTestData,
    CsvTestData,
    UploadFileFactory,
)

EXPECTED_ROW_COUNT = 2


@pytest.mark.anyio
async def test_process_csv_file_success(
    mock_valid_csv_data: CsvTestData,
    make_upload_file: UploadFileFactory,
) -> None:
    """Test successful processing of various CSV file formats."""
    mock_file = make_upload_file(
        filename=mock_valid_csv_data.filename,
        content=mock_valid_csv_data.csv_data.encode("utf-8"),
    )

    result = await process_csv_file(mock_file)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == EXPECTED_ROW_COUNT
    assert list(result.columns) == mock_valid_csv_data.expected_columns
    mock_file.read.assert_called_once()


@pytest.mark.anyio
async def test_process_csv_file_invalid_filename(mock_invalid_file: Mock) -> None:
    """Test handling of invalid filenames."""
    with pytest.raises(CsvFormatError) as exc_info:
        await process_csv_file(mock_invalid_file)

    assert "Invalid file type" in str(exc_info.value)
    mock_invalid_file.read.assert_not_called()


@pytest.mark.anyio
async def test_process_csv_file_error_scenarios(
    mock_error_csv_scenarios: CsvErrorTestData,
    make_upload_file: UploadFileFactory,
) -> None:
    """Test various CSV error scenarios using parametrized fixture."""
    mock_file = make_upload_file(
        filename=mock_error_csv_scenarios.filename,
        content=mock_error_csv_scenarios.read_value,
    )

    with pytest.raises(mock_error_csv_scenarios.expected_exception_type) as exc_info:
        await process_csv_file(mock_file)

    assert mock_error_csv_scenarios.expected_message in str(exc_info.value)


@pytest.mark.anyio
async def test_process_csv_file_insufficient_columns(
    make_upload_file: UploadFileFactory,
) -> None:
    """Test CSV with only one column is rejected."""
    mock_file = make_upload_file(
        filename="single_column.csv",
        content=b"single_column\nvalue1\nvalue2",
    )

    with pytest.raises(CsvContentError) as exc_info:
        await process_csv_file(mock_file)

    assert "CSV must contain at least two columns" in str(exc_info.value)


@pytest.mark.anyio
async def test_process_csv_file_skips_target_validation_when_not_required(
    make_upload_file: UploadFileFactory,
) -> None:
    mock_file = make_upload_file(
        filename="features_only.csv",
        content=b"feature_a,feature_b\n1,2\n3,4",
    )

    result = await process_csv_file(mock_file, require_target=False)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature_a", "feature_b"]


@pytest.mark.anyio
async def test_process_csv_file_rejects_headers_only(
    make_upload_file: UploadFileFactory,
) -> None:
    """Test CSV file with headers but no data rows is rejected."""
    mock_file = make_upload_file(
        filename="headers_only.csv",
        content=b"name,age,city",
    )

    with pytest.raises(CsvContentError) as exc_info:
        await process_csv_file(mock_file)

    assert "CSV contains no data rows" in str(exc_info.value)


@pytest.mark.parametrize(
    "size_bytes",
    [
        pytest.param(50 * 1024 * 1024 + 1, id="just_over_limit"),
        pytest.param(200 * 1024 * 1024, id="way_over_limit"),
    ],
)
@pytest.mark.anyio
async def test_process_csv_file_rejects_oversize(
    size_bytes: int,
    make_upload_file: UploadFileFactory,
) -> None:
    """Test CSV files exceeding the 50 MB cap raise CsvSizeError."""
    mock_file = make_upload_file(filename="huge.csv", size=size_bytes)

    with pytest.raises(CsvSizeError) as exc_info:
        await process_csv_file(mock_file)

    assert "File too large" in str(exc_info.value)
