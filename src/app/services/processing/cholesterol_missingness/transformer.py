from pandas import DataFrame

from app.services.processing.base import BasePreprocessor

from .config import CholesterolMissingnessConfig


class CholesterolMissingness(BasePreprocessor):
    """Creates a flag indicating whether it's missing."""

    def __init__(self) -> None:
        super().__init__()
        self.config = CholesterolMissingnessConfig()

    def _add_missingness_flag(self, d: DataFrame) -> DataFrame:
        """
        Args:
            d: Input DataFrame.

        Returns:
            DataFrame with flag added.
        """
        df = d.copy()
        df[self.config.output_column] = df[self.config.source_column].isna().astype(int)
        return df

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Args:
            X: Input DataFrame containing the cholesterol column.

        Returns:
            DataFrame with missingness flag added; source column is retained.
        """
        return X.pipe(self._add_missingness_flag)
