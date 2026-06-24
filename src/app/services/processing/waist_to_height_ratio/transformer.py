import numpy as np
from pandas import DataFrame

from app.services.processing.base import BasePreprocessor
from app.services.processing.exceptions import MissingFeatureColumnsError

from .config import WaistToHeightRatioConfig


class WaistToHeightRatio(BasePreprocessor):
    """Computes waist-to-height ratio and drops source columns."""

    def __init__(self) -> None:
        super().__init__()
        self.config = WaistToHeightRatioConfig()

    def _validate_columns_exist(self, d: DataFrame) -> None:
        """Validate that required columns exist in the DataFrame.
        Rows with negative or missing height produce NaN ratios.

        Args:
            d: Input DataFrame to validate.

        Raises:
            MissingFeatureColumnsError: If any required column is missing.
        """
        required = {self.config.waist_column, self.config.height_column}
        missing = sorted(required - set(d.columns))
        if missing:
            raise MissingFeatureColumnsError(missing)

    def _compute_ratio(self, d: DataFrame) -> DataFrame:
        """Add waist-to-height ratio column.

        Args:
            d: Input DataFrame containing waist and height columns.

        Returns:
            Augmented DataFrame with ratio column added.
        """
        df = d.copy()
        height = df[self.config.height_column].where(
            df[self.config.height_column] > 0, np.nan
        )

        df[self.config.output_column] = df[self.config.waist_column] / height
        return df

    def transform(self, X: DataFrame) -> DataFrame:
        """Compute waist-to-height ratio from raw anthropometric columns.

        Args:
            X: Input DataFrame containing waist and height columns.

        Returns:
            DataFrame with ratio column added and source columns removed.
        """
        self._validate_columns_exist(X)

        return X.pipe(self._compute_ratio).drop(
            columns=[self.config.waist_column, self.config.height_column]
        )
