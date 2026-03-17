from typing import cast

import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from app.services.processing.scaler.config import BINARY_COLS, CONTINUOUS_COLS
from app.services.processing.service import ProcessingService

MEAN_TOLERANCE: float = 1e-10
EXPECTED_STD: float = 1.0


def test_pipeline_output_columns(pipeline_dataframe: pd.DataFrame) -> None:
    result = cast(
        "pd.DataFrame",
        ProcessingService().pipeline.fit_transform(pipeline_dataframe),  # pyright: ignore[reportUnknownMemberType]
    )

    assert list(result.columns) == CONTINUOUS_COLS + BINARY_COLS


def test_pipeline_no_nans_in_output(
    pipeline_dataframe_with_nans: pd.DataFrame,
) -> None:
    pipeline = ProcessingService().pipeline
    result = cast(
        "pd.DataFrame",
        pipeline.fit_transform(pipeline_dataframe_with_nans),  # pyright: ignore[reportUnknownMemberType]
    )

    assert result.isna().sum().sum() == 0


def test_pipeline_binary_columns_contain_only_zero_or_one(
    pipeline_dataframe: pd.DataFrame,
) -> None:
    result = cast(
        "pd.DataFrame",
        ProcessingService().pipeline.fit_transform(pipeline_dataframe),  # pyright: ignore[reportUnknownMemberType]
    )

    for col in BINARY_COLS:
        assert set(result[col].unique()).issubset({0, 1})


def test_pipeline_continuous_columns_scaled(
    pipeline_dataframe: pd.DataFrame,
    continuous_col: str,
) -> None:
    result = cast(
        "pd.DataFrame",
        ProcessingService().pipeline.fit_transform(pipeline_dataframe),  # pyright: ignore[reportUnknownMemberType]
    )

    assert abs(result[continuous_col].mean()) < MEAN_TOLERANCE
    assert abs(result[continuous_col].std(ddof=0) - EXPECTED_STD) < MEAN_TOLERANCE


def test_pipeline_fit_transform_separation(
    pipeline_fit_dataframe: pd.DataFrame,
    pipeline_transform_dataframe: pd.DataFrame,
) -> None:
    # fit: diastolic_bp = [60, 80, 100, 80] → mean = 80
    # transform: diastolic_bp = [80, 80, 80, 80] — exactly the fit mean → scaled = 0.0
    pipeline = ProcessingService().pipeline
    pipeline.fit(pipeline_fit_dataframe)  # pyright: ignore[reportUnknownMemberType]
    result = cast(
        "pd.DataFrame",
        pipeline.transform(pipeline_transform_dataframe),  # pyright: ignore[reportUnknownMemberType]
    )

    expected = pytest.approx([0.0, 0.0, 0.0, 0.0])  # pyright: ignore[reportUnknownMemberType]
    assert result["diastolic_bp"].tolist() == expected


def test_pipeline_snapshot(
    pipeline_dataframe: pd.DataFrame,
    snapshot: SnapshotAssertion,
) -> None:
    result = cast(
        "pd.DataFrame",
        ProcessingService().pipeline.fit_transform(pipeline_dataframe),  # pyright: ignore[reportUnknownMemberType]
    )

    assert result.to_csv().strip() == snapshot
