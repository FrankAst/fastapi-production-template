import pandas as pd
import pytest

from app.services.processing.scaler.config import BINARY_COLS, CONTINUOUS_COLS
from app.services.processing.scaler.transformer import Scaler

MEAN_TOLERANCE: float = 1e-10
EXPECTED_STD: float = 1.0


def test_binary_columns_unchanged(scaler_dataframe: pd.DataFrame) -> None:
    result = Scaler().fit(scaler_dataframe).transform(scaler_dataframe)

    for col in BINARY_COLS:
        assert result[col].tolist() == scaler_dataframe[col].tolist()


def test_continuous_columns_scaled(
    scaler_dataframe: pd.DataFrame,
    continuous_col: str,
) -> None:
    result = Scaler().fit(scaler_dataframe).transform(scaler_dataframe)

    assert abs(result[continuous_col].mean()) < MEAN_TOLERANCE
    assert abs(result[continuous_col].std(ddof=0) - EXPECTED_STD) < MEAN_TOLERANCE


def test_fit_transform_separation(
    scaler_fit_dataframe: pd.DataFrame,
    scaler_transform_dataframe: pd.DataFrame,
) -> None:
    # fit data: diastolic_bp = [60, 80, 100] → mean = 80
    # transform data: diastolic_bp = [80, 80, 80]
    # each value equals the fit mean → scaled output should be 0.0
    scaler = Scaler()
    scaler.fit(scaler_fit_dataframe)
    result = scaler.transform(scaler_transform_dataframe)

    assert result["diastolic_bp"].tolist() == pytest.approx([0.0, 0.0, 0.0])  # pyright: ignore[reportUnknownMemberType]


def test_column_names_preserved(scaler_dataframe: pd.DataFrame) -> None:
    result = Scaler().fit(scaler_dataframe).transform(scaler_dataframe)

    assert list(result.columns) == CONTINUOUS_COLS + BINARY_COLS


def test_output_is_dataframe(scaler_dataframe: pd.DataFrame) -> None:
    result = Scaler().fit(scaler_dataframe).transform(scaler_dataframe)

    assert isinstance(result, pd.DataFrame)
