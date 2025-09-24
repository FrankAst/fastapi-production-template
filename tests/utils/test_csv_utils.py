"""Tests for CSV processing utilities."""

import pytest
from fastapi import HTTPException, UploadFile
from io import BytesIO
import pandas as pd
from unittest.mock import AsyncMock, Mock

from app.utils.csv_utils import (
    validate_csv_filename,
    read_csv_content,
    parse_csv_to_dataframe,
    process_csv_file,
)


class TestValidateCsvFilename:
    """Test cases for validate_csv_filename function."""

    def test_validate_csv_filename_success(self):
        """Test that valid CSV filenames pass validation."""
        # Should not raise any exception
        validate_csv_filename("test.csv")
        validate_csv_filename("data_file.csv")
        validate_csv_filename("file_with_underscores.csv")

    def test_validate_csv_filename_fails_for_non_csv(self):
        """Test that non-CSV filenames raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_csv_filename("test.txt")
        assert exc_info.value.status_code == 400
        assert "Only CSV files are supported" in str(exc_info.value.detail)

        with pytest.raises(HTTPException) as exc_info:
            validate_csv_filename("test.xlsx")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            validate_csv_filename("test.json")
        assert exc_info.value.status_code == 400

    def test_validate_csv_filename_fails_for_none(self):
        """Test that None filename raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_csv_filename(None)
        assert exc_info.value.status_code == 400
        assert "Only CSV files are supported" in str(exc_info.value.detail)

    def test_validate_csv_filename_fails_for_empty_string(self):
        """Test that empty filename raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_csv_filename("")
        assert exc_info.value.status_code == 400


class TestReadCsvContent:
    """Test cases for read_csv_content function."""

    @pytest.mark.anyio
    async def test_read_csv_content_success(self):
        """Test successful reading of CSV content."""
        # Create mock UploadFile
        csv_data = "name,age\nJohn,25\nJane,30"
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

        result = await read_csv_content(mock_file)
        
        assert result == csv_data
        mock_file.read.assert_called_once()

    @pytest.mark.anyio
    async def test_read_csv_content_handles_decode_error(self):
        """Test handling of file decode errors."""
        # Create mock file that returns invalid bytes
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b'\x80\x81\x82')  # Invalid UTF-8

        with pytest.raises(HTTPException) as exc_info:
            await read_csv_content(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "Error reading file content" in str(exc_info.value.detail)

    @pytest.mark.anyio
    async def test_read_csv_content_handles_read_error(self):
        """Test handling of file read errors."""
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(side_effect=Exception("File read error"))

        with pytest.raises(HTTPException) as exc_info:
            await read_csv_content(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "Error reading file content" in str(exc_info.value.detail)


class TestParseCsvToDataframe:
    """Test cases for parse_csv_to_dataframe function."""

    def test_parse_csv_to_dataframe_success(self):
        """Test successful parsing of CSV content to DataFrame."""
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA"
        
        result = parse_csv_to_dataframe(csv_content)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age", "city"]
        assert result.iloc[0]["name"] == "John"
        assert result.iloc[0]["age"] == 25
        assert result.iloc[1]["name"] == "Jane"

    def test_parse_csv_to_dataframe_empty_content(self):
        """Test parsing empty CSV content raises HTTPException."""
        csv_content = ""
        
        with pytest.raises(HTTPException) as exc_info:
            parse_csv_to_dataframe(csv_content)
        
        assert exc_info.value.status_code == 400
        assert "Error parsing CSV" in str(exc_info.value.detail)

    def test_parse_csv_to_dataframe_handles_malformed_csv(self):
        """Test handling of malformed CSV content."""
        # CSV with inconsistent columns
        csv_content = "name,age\nJohn,25,extra_column\nJane"
        
        # This should still work as pandas is quite forgiving
        result = parse_csv_to_dataframe(csv_content)
        assert isinstance(result, pd.DataFrame)

    def test_parse_csv_to_dataframe_handles_parse_error(self):
        """Test handling of severe CSV parsing errors."""
        # Create content that will cause pandas to fail
        csv_content = None
        
        with pytest.raises(HTTPException) as exc_info:
            parse_csv_to_dataframe(csv_content)
        
        assert exc_info.value.status_code == 400
        assert "Error parsing CSV" in str(exc_info.value.detail)


class TestProcessCsvFile:
    """Test cases for process_csv_file function (complete pipeline)."""

    @pytest.mark.anyio
    async def test_process_csv_file_complete_pipeline_success(self):
        """Test the complete CSV processing pipeline."""
        csv_data = "name,age\nJohn,25\nJane,30"
        
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.csv"
        mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

        result = await process_csv_file(mock_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age"]
        mock_file.read.assert_called_once()

    @pytest.mark.anyio
    async def test_process_csv_file_invalid_filename(self):
        """Test pipeline fails with invalid filename."""
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.txt"

        with pytest.raises(HTTPException) as exc_info:
            await process_csv_file(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "Only CSV files are supported" in str(exc_info.value.detail)

    @pytest.mark.anyio
    async def test_process_csv_file_read_error(self):
        """Test pipeline handles read errors."""
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.csv"
        mock_file.read = AsyncMock(side_effect=Exception("Read failed"))

        with pytest.raises(HTTPException) as exc_info:
            await process_csv_file(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "Error reading file content" in str(exc_info.value.detail)

    @pytest.mark.anyio
    async def test_process_csv_file_with_complex_data(self):
        """Test pipeline with more complex CSV data."""
        csv_data = "id,name,score,active\n1,Alice,95.5,true\n2,Bob,87.2,false"
        
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "complex_data.csv"
        mock_file.read = AsyncMock(return_value=csv_data.encode("utf-8"))

        result = await process_csv_file(mock_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name", "score", "active"]
        assert result.iloc[0]["name"] == "Alice"
        assert result.iloc[0]["score"] == 95.5