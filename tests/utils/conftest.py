from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from fastapi import UploadFile

# Fixtures for utility tests will be added here as needed


@dataclass
class CsvTestData:
    """Test data structure for CSV processing tests."""

    csv_data: str
    expected_columns: list[str]
    filename: str


@dataclass
class CsvErrorTestData:
    """Test data structure for CSV error scenarios."""

    filename: str
    read_value: bytes
    expected_message: str


@pytest.fixture(
    params=[
        "test.txt",
        "test.xlsx",
        "test.json",
        None,
        "",
        "test.pdf",
        "test.docx",
        "test",
    ]
)
def mock_invalid_file(request: pytest.FixtureRequest) -> Mock:
    """
    Fixture to create a mock UploadFile with various invalid filenames.

    Returns:
        Mock: A mock UploadFile instance with an invalid filename.
    """
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = request.param
    return mock_file


@pytest.fixture(
    params=[
        CsvTestData(
            csv_data="name,age\nJohn,25\nJane,30",
            expected_columns=["name", "age"],
            filename="basic.csv",
        ),
        CsvTestData(
            csv_data="id,name,score,active\n1,Alice,95.5,true\n2,Bob,87.2,false",
            expected_columns=["id", "name", "score", "active"],
            filename="complex.csv",
        ),
    ]
)
def mock_valid_csv_data(request: pytest.FixtureRequest) -> CsvTestData:
    """
    Fixture providing various valid CSV data scenarios.
    Returns:
        CsvTestData: A dataclass containing CSV data, expected columns, and filename.
    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        CsvErrorTestData(
            filename="decode_error.csv",
            read_value=b"\x80\x81\x82",  # Invalid UTF-8
            expected_message="Error reading file content",
        ),
        CsvErrorTestData(
            filename="empty.csv",
            read_value=b"",
            expected_message="Error parsing CSV",
        ),
        CsvErrorTestData(
            filename="malformed.csv",
            read_value=b"name,age\nJohn,25,extra_column\nJane",
            expected_message="Target column (last column) contains missing values",
        ),
        CsvErrorTestData(
            filename="nan_target.csv",
            read_value=b"feature1,feature2,target\n25.0,10.5,5.0\n30.0,15.2,6.0\n35.0,20.1,",
            expected_message="Target column (last column) contains missing values",
        ),
    ]
)
def mock_error_csv_scenarios(request: pytest.FixtureRequest) -> CsvErrorTestData:
    """
    Fixture providing various CSV error scenarios.
    Returns:
        CsvErrorTestData: A dataclass containing error scenario data.
    """
    return request.param  # type: ignore[no-any-return]
