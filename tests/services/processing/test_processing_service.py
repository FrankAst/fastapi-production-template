import pandas as pd
from syrupy.assertion import SnapshotAssertion

from app.services.processing import AgeBinner


def test_creates_expected_columns(
    snapshot: SnapshotAssertion,
    valid_training_dataframe: pd.DataFrame,
    age_categories: list[str],
) -> None:

    result = AgeBinner().transform(valid_training_dataframe)

    assert set(age_categories).issubset(result.columns)
    assert result.to_csv().strip() == snapshot


def test_removes_original_column(valid_training_dataframe: pd.DataFrame) -> None:

    result = AgeBinner().transform(valid_training_dataframe)

    assert "RIDAGEYR" not in result.columns


def test_age_maps_to_correct_category(
    age_with_expected_category: tuple[float | None, str],
    age_categories: list[str],
) -> None:

    age, expected_category = age_with_expected_category
    df = pd.DataFrame({"RIDAGEYR": [age], "feature": [1]})
    result = AgeBinner().transform(df)

    for category in age_categories:
        expected = 1 if category == expected_category else 0
        actual = result[category].iloc[0]
        assert actual == expected


def test_mutual_exclusivity(
    dataframe_with_all_age_categories: pd.DataFrame,
    age_categories: list[str],
) -> None:

    result = AgeBinner().transform(dataframe_with_all_age_categories)

    row_sums = result[age_categories].sum(axis=1)
    assert (row_sums == 1).all(), "Each row should have exactly one age category"
