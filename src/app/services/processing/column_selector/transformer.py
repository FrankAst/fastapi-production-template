from pandas import DataFrame

from app.services.processing.base import BasePreprocessor

from .config import SELECTED_FEATURES


class ColumnSelector(BasePreprocessor):
    """Selects and reorders the 13 features expected by the model."""

    @staticmethod
    def _validate_columns(d: DataFrame) -> None:
        """Raise if any expected feature is missing from the DataFrame.

        Args:
            d: Input DataFrame to validate.

        Raises:
            ValueError: Lists every missing column.
        """
        missing = [col for col in SELECTED_FEATURES if col not in d.columns]
        if missing:
            msg = f"Missing expected columns: {missing}"
            raise ValueError(msg)

    def transform(self, X: DataFrame) -> DataFrame:
        """Select and reorder columns to match the expected feature set.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with exactly the 13 model features in canonical order.
        """
        self._validate_columns(X)
        return X[list(SELECTED_FEATURES)]
