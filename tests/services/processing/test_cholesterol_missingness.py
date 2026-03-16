"""Tests for CholesterolMissingness transformer."""

import pandas as pd
import pytest

from app.services.processing.cholesterol_missingness.transformer import (
    CholesterolMissingness,
)


def test_creates_missingness_flag(cholesterol_dataframe: pd.DataFrame) -> None:
    result = CholesterolMissingness().transform(cholesterol_dataframe)

    assert result["told_high_cholesterol_missing"].tolist() == [0, 0, 1, 0]


def test_retains_source_column(cholesterol_dataframe: pd.DataFrame) -> None:
    result = CholesterolMissingness().transform(cholesterol_dataframe)

    assert "told_high_cholesterol" in result.columns
    # NaN positions should remain NaN — the transformer does not impute
    assert pd.isna(result["told_high_cholesterol"].iloc[2])
    assert result["told_high_cholesterol"].iloc[0] == pytest.approx(1.0)
    assert result["told_high_cholesterol"].iloc[1] == pytest.approx(0.0)


def test_preserves_other_columns(cholesterol_dataframe: pd.DataFrame) -> None:
    result = CholesterolMissingness().transform(cholesterol_dataframe)

    assert "other_col" in result.columns
    assert result["other_col"].tolist() == cholesterol_dataframe["other_col"].tolist()


def test_all_nan_produces_all_ones(all_nan_cholesterol_dataframe: pd.DataFrame) -> None:
    result = CholesterolMissingness().transform(all_nan_cholesterol_dataframe)

    assert result["told_high_cholesterol_missing"].tolist() == [1, 1, 1]


def test_no_nan_produces_all_zeros(no_nan_cholesterol_dataframe: pd.DataFrame) -> None:
    result = CholesterolMissingness().transform(no_nan_cholesterol_dataframe)

    assert result["told_high_cholesterol_missing"].tolist() == [0, 0, 0, 0]


def test_raises_on_missing_column() -> None:
    df = pd.DataFrame({"other_col": [1, 2]})

    with pytest.raises(KeyError):
        CholesterolMissingness().transform(df)
