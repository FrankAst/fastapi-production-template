import pandas as pd

from app.services.processing.service import ProcessingService


def test_skip_true_condition(valid_training_dataframe: pd.DataFrame) -> None:
    """
    Test that if skip == true condition, dataframe is returned as-is.
    """

    processed_df = ProcessingService.preprocess(
        valid_training_dataframe, skip=True, training=True
    )
    assert processed_df.equals(valid_training_dataframe), (
        "DataFrame should be unchanged when skip=True"
    )


def test_target_column_position_preserved(
    valid_training_dataframe: pd.DataFrame,
) -> None:
    """
    Test that the target column remains the last column after preprocessing.
    """
    target = valid_training_dataframe.columns[-1]

    processed_df = ProcessingService.preprocess(
        valid_training_dataframe, skip=False, training=True
    )
    assert processed_df.columns[-1] == target, (
        "Target column should remain the last column after preprocessing"
    )


# =============================================================================
# Age Binning Tests
# =============================================================================

# Row indices for df_age_binning_test_cases fixture
# Age semantics:
# - 18-44  -> young_adult
# - 45-64  -> middle_age
# - 65-79  -> senior
# - 80     -> elderly (top-coded)
# - NaN    -> age_unknown
YOUNG_ADULT_ROWS = slice(0, 4)  # ages: 18, 25, 30, 44
MIDDLE_AGE_ROWS = slice(4, 8)  # ages: 45, 50, 55, 64
SENIOR_ROWS = slice(8, 12)  # ages: 65, 70, 75, 79
ELDERLY_ROWS = slice(12, 14)  # ages: 80, 80 (top-coded)
UNKNOWN_ROWS = slice(14, 16)  # ages: NaN, NaN

# Boundary indices (first age of each group)
BOUNDARY_YOUNG_ADULT = 0  # age 18
BOUNDARY_MIDDLE_AGE = 4  # age 45
BOUNDARY_SENIOR = 8  # age 65


class TestAgeBinningColumns:
    """Tests for verifying age binning column creation and removal."""

    @staticmethod
    def test_creates_expected_columns(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test that age binning creates the expected one-hot encoded columns."""
        result = ProcessingService.age_binning(df_age_binning_test_cases.copy())

        expected_columns = {
            "young_adult",
            "middle_age",
            "senior",
            "elderly",
            "age_unknown",
        }
        assert expected_columns.issubset(result.columns)

    @staticmethod
    def test_removes_original_columns(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test that original age columns are removed after binning."""
        result = ProcessingService.age_binning(df_age_binning_test_cases.copy())

        assert not {"RIDAGEYR", "age_group"}.issubset(result.columns)


class TestAgeBinningCategories:
    """Tests for verifying correct age categorization."""

    @staticmethod
    def test_young_adult_range(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test ages 18-44 are categorized as young_adult."""
        df = df_age_binning_test_cases.iloc[YOUNG_ADULT_ROWS].copy()
        result = ProcessingService.age_binning(df)

        assert all(result["young_adult"] == 1), "All ages 18-44 should be young_adult"
        assert all(result["middle_age"] == 0), "No ages 18-44 should be middle_age"
        assert all(result["senior"] == 0), "No ages 18-44 should be senior"
        assert all(result["elderly"] == 0), "No ages 18-44 should be elderly"
        assert all(result["age_unknown"] == 0), "No ages 18-44 should be age_unknown"

    @staticmethod
    def test_middle_age_range(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test ages 45-64 are categorized as middle_age."""
        df = df_age_binning_test_cases.iloc[MIDDLE_AGE_ROWS].copy()
        result = ProcessingService.age_binning(df)

        assert all(result["young_adult"] == 0), "No ages 45-64 should be young_adult"
        assert all(result["middle_age"] == 1), "All ages 45-64 should be middle_age"
        assert all(result["senior"] == 0), "No ages 45-64 should be senior"
        assert all(result["elderly"] == 0), "No ages 45-64 should be elderly"
        assert all(result["age_unknown"] == 0), "No ages 45-64 should be age_unknown"

    @staticmethod
    def test_senior_range(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test ages 65-79 are categorized as senior."""
        df = df_age_binning_test_cases.iloc[SENIOR_ROWS].copy()
        result = ProcessingService.age_binning(df)

        assert all(result["young_adult"] == 0), "No ages 65-79 should be young_adult"
        assert all(result["middle_age"] == 0), "No ages 65-79 should be middle_age"
        assert all(result["senior"] == 1), "All ages 65-79 should be senior"
        assert all(result["elderly"] == 0), "No ages 65-79 should be elderly"
        assert all(result["age_unknown"] == 0), "No ages 65-79 should be age_unknown"

    @staticmethod
    def test_elderly_top_coded(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test age 80 (top-coded) is categorized as elderly."""
        df = df_age_binning_test_cases.iloc[ELDERLY_ROWS].copy()
        result = ProcessingService.age_binning(df)

        assert all(result["young_adult"] == 0), "Age 80 should not be young_adult"
        assert all(result["middle_age"] == 0), "Age 80 should not be middle_age"
        assert all(result["senior"] == 0), "Age 80 should not be senior"
        assert all(result["elderly"] == 1), "Age 80 should be elderly"
        assert all(result["age_unknown"] == 0), "Age 80 should not be age_unknown"

    @staticmethod
    def test_unknown_nan_ages(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test NaN ages are categorized as age_unknown."""
        df = df_age_binning_test_cases.iloc[UNKNOWN_ROWS].copy()
        result = ProcessingService.age_binning(df)

        assert all(result["young_adult"] == 0), "NaN ages should not be young_adult"
        assert all(result["middle_age"] == 0), "NaN ages should not be middle_age"
        assert all(result["senior"] == 0), "NaN ages should not be senior"
        assert all(result["elderly"] == 0), "NaN ages should not be elderly"
        assert all(result["age_unknown"] == 1), "NaN ages should be age_unknown"


class TestAgeBinningBoundariesAndEdgeCases:
    """Tests for boundary conditions and edge cases."""

    @staticmethod
    def test_exact_boundary_values(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test ages exactly at bin boundaries (18, 45, 65)."""
        df = df_age_binning_test_cases.iloc[
            [BOUNDARY_YOUNG_ADULT, BOUNDARY_MIDDLE_AGE, BOUNDARY_SENIOR]
        ].copy()
        result = ProcessingService.age_binning(df)

        # Age 18 -> young_adult (first row)
        assert result.iloc[0]["young_adult"] == 1
        assert result.iloc[0]["middle_age"] == 0
        assert result.iloc[0]["senior"] == 0

        # Age 45 -> middle_age (second row)
        assert result.iloc[1]["young_adult"] == 0
        assert result.iloc[1]["middle_age"] == 1
        assert result.iloc[1]["senior"] == 0

        # Age 65 -> senior (third row)
        assert result.iloc[2]["young_adult"] == 0
        assert result.iloc[2]["middle_age"] == 0
        assert result.iloc[2]["senior"] == 1

    @staticmethod
    def test_mutual_exclusivity(
        df_age_binning_test_cases: pd.DataFrame,
    ) -> None:
        """Test that each row has exactly one age category set to 1."""
        result = ProcessingService.age_binning(df_age_binning_test_cases.copy())

        age_columns = ["young_adult", "middle_age", "senior", "elderly", "age_unknown"]
        row_sums = result[age_columns].sum(axis=1)

        assert all(row_sums == 1), "Each row should have exactly one age category"
