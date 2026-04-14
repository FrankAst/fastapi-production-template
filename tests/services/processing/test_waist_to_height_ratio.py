import pandas as pd
import pytest

from app.services.processing.waist_to_height_ratio.transformer import WaistToHeightRatio


def test_computes_correct_ratio(waist_height_dataframe: pd.DataFrame) -> None:
    result = WaistToHeightRatio().transform(waist_height_dataframe)

    expected = pytest.approx([0.5, 0.6])  # pyright: ignore[reportUnknownMemberType]
    assert result["waist_to_height_ratio"].tolist() == expected


def test_drops_source_columns(waist_height_dataframe: pd.DataFrame) -> None:
    result = WaistToHeightRatio().transform(waist_height_dataframe)

    assert "BMXWAIST" not in result.columns
    assert "BMXHT" not in result.columns


def test_preserves_other_columns(waist_height_dataframe: pd.DataFrame) -> None:
    result = WaistToHeightRatio().transform(waist_height_dataframe)

    assert "other_col" in result.columns
    assert result["other_col"].tolist() == waist_height_dataframe["other_col"].tolist()


def test_edge_case_produces_expected_ratio(
    edge_case_waist_height: tuple[float | None, float | None, float | None],
) -> None:
    waist, height, expected = edge_case_waist_height
    df = pd.DataFrame({"BMXWAIST": [waist], "BMXHT": [height]})

    result = WaistToHeightRatio().transform(df)

    ratio = result["waist_to_height_ratio"].iloc[0]
    if expected is None:
        assert pd.isna(ratio)
    else:
        assert ratio == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test_raises_on_missing_column() -> None:
    df = pd.DataFrame({"other_col": [1, 2]})

    with pytest.raises(ValueError, match="requires columns"):
        WaistToHeightRatio().transform(df)
