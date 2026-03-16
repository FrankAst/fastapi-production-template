from pandas import DataFrame

from app.services.processing.base import BasePreprocessor

from .config import ColumnSelectorConfig


class ColumnSelector(BasePreprocessor):
    """Selects and reorders the 13 features expected by the LR VIF+BIC model."""

    def __init__(self) -> None:
        super().__init__()
        self.config = ColumnSelectorConfig()

    def _validate_columns(self, d: DataFrame) -> None:
        """Raise if any expected feature column is missing from the DataFrame.

        Args:
            d: Input DataFrame to validate.

        Raises:
            ValueError: Lists every missing column so the caller can
            diagnose upstream gaps.
        """
        missing = [col for col in self.config.features if col not in d.columns]
        if missing:
            msg = f"Missing expected columns: {missing}"
            raise ValueError(msg)

    def transform(self, X: DataFrame) -> DataFrame:
        """Select and reorder columns to match the model's expected feature set.

        Args:
            X: Input DataFrame containing at least all expected feature columns.

        Returns:
            DataFrame with exactly the 13 model features in canonical order.
        """
        self._validate_columns(X)
        return X[self.config.features]
