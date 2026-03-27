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
            csv_data="feature1,has_diabetes_or_prediabetes\n1.0,0.0\n2.0,1.0",
            expected_columns=["feature1", "has_diabetes_or_prediabetes"],
            filename="basic.csv",
        ),
        CsvTestData(
            csv_data="feature1,feature2,feature3,has_diabetes_or_prediabetes\n1.0,2.0,3.0,0.0\n4.0,5.0,6.0,1.0",
            expected_columns=[
                "feature1",
                "feature2",
                "feature3",
                "has_diabetes_or_prediabetes",
            ],
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
            read_value=b"feature1,has_diabetes_or_prediabetes\n1.0,0.0\n2.0,",
            expected_message=(
                "Target column 'has_diabetes_or_prediabetes' contains missing values"
            ),
        ),
        CsvErrorTestData(
            filename="nan_target.csv",
            read_value=b"feature1,feature2,has_diabetes_or_prediabetes\n25.0,10.5,0.0\n30.0,15.2,1.0\n35.0,20.1,",
            expected_message=(
                "Target column 'has_diabetes_or_prediabetes' contains missing values"
            ),
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
