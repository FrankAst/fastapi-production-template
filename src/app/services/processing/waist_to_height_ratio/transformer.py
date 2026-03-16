import numpy as np
from pandas import DataFrame

from app.services.processing.base import BasePreprocessor

from .config import WaistToHeightRatioConfig


class WaistToHeightRatio(BasePreprocessor):
    """Computes waist-to-height ratio and drops source columns."""

    def __init__(self) -> None:
        super().__init__()
        self.config = WaistToHeightRatioConfig()

    def _compute_ratio(self, d: DataFrame) -> DataFrame:
        """Add waist-to-height ratio column; guard against zero-height division.

        Args:
            d: Input DataFrame containing waist and height columns.

        Returns:
            DataFrame with ratio column added (source columns still present).
        """
        df = d.copy()
        height = df[self.config.height_column].replace(0, np.nan)
        df[self.config.output_column] = df[self.config.waist_column] / height
        return df

    def transform(self, X: DataFrame) -> DataFrame:
        """Compute waist-to-height ratio from raw anthropometric columns.

        Args:
            X: Input DataFrame containing waist and height columns.

        Returns:
            DataFrame with ratio column added and source columns removed.
        """
        return X.pipe(self._compute_ratio).drop(
            columns=[self.config.waist_column, self.config.height_column]
        )
