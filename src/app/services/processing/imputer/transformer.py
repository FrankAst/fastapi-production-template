from typing import Self, cast

from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from app.services.processing.base import BasePreprocessor

from .config import ImputerConfig


class Imputer(BasePreprocessor):
    """Stateful imputer applying per-column-group strategies via ColumnTransformer.

    Strategy:
        - Zero-fill: columns where NaN means "doesn't do this activity"
        - Median-fill: columns with structural or random missingness
        - Passthrough: engineered binary flags that are already complete
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = ImputerConfig()
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "zero_fill",
                    SimpleImputer(strategy="constant", fill_value=0),
                    self.config.zero_fill_cols,
                ),
                (
                    "median_fill",
                    SimpleImputer(strategy="median"),
                    self.config.median_fill_cols,
                ),
                ("passthrough", "passthrough", self.config.passthrough_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._column_transformer.set_output(transform="pandas")

    def fit(self, X: DataFrame, y: Series | None = None) -> Self:
        """Learn fill values from training data.

        Args:
            X: Training DataFrame with all 13 model features.
            y: Ignored; present for sklearn pipeline compatibility.

        Returns:
            Self: The fitted imputer instance.
        """
        self._column_transformer.fit(X, y)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Apply learned imputation to data.

        Args:
            X: DataFrame with all 13 model features.

        Returns:
            DataFrame with no NaN values and columns in canonical order.
        """
        return cast("DataFrame", self._column_transformer.transform(X))
