from pandas import DataFrame

from app.services.processing.base import BasePreprocessor

from .config import CholesterolMissingnessConfig


class CholesterolMissingness(BasePreprocessor):
    """Creates a binary flag indicating whether cholesterol value is missing."""

    def __init__(self) -> None:
        super().__init__()
        self.config = CholesterolMissingnessConfig()

    def _add_missingness_flag(self, d: DataFrame) -> DataFrame:
        """Add binary missingness indicator column.

        Args:
            d: Input DataFrame containing the cholesterol column.

        Returns:
            DataFrame with missingness flag added; source column is retained.
        """
        df = d.copy()
        df[self.config.output_column] = df[self.config.source_column].isna().astype(int)
        return df

    def transform(self, X: DataFrame) -> DataFrame:
        """Add a missingness indicator for the cholesterol column.

        Args:
            X: Input DataFrame containing the cholesterol column.

        Returns:
            DataFrame with missingness flag added; source column is retained.
        """
        return X.pipe(self._add_missingness_flag)
