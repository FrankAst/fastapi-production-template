import pandas as pd
import pytest

from app.services.processing.column_selector.config import SELECTED_FEATURES
from app.services.processing.column_selector.transformer import ColumnSelector


def test_selects_correct_features(post_fe_dataframe: pd.DataFrame) -> None:
    result = ColumnSelector().transform(post_fe_dataframe)

    assert list(result.columns) == SELECTED_FEATURES
    assert result.shape[1] == len(SELECTED_FEATURES)


def test_drops_extra_columns(post_fe_dataframe: pd.DataFrame) -> None:
    result = ColumnSelector().transform(post_fe_dataframe)

    assert "middle_age" not in result.columns
    assert "senior" not in result.columns
    assert "age_unknown" not in result.columns


def test_enforces_column_order(shuffled_post_fe_dataframe: pd.DataFrame) -> None:
    result = ColumnSelector().transform(shuffled_post_fe_dataframe)

    assert list(result.columns) == SELECTED_FEATURES


def test_raises_on_missing_feature() -> None:
    df = pd.DataFrame({"young_adult": [1], "elderly": [0]})  # missing 11 features

    with pytest.raises(ValueError, match="Missing expected columns"):
        ColumnSelector().transform(df)
