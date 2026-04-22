from abc import abstractmethod
from typing import Self, cast

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all preprocessing transformers."""

    def __init__(self) -> None:
        self._is_fitted: bool = False

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201  # pylint: disable=bad-dunder-name
        """Return True if fit() has been called on this transformer.

        Returns:
            bool: True after fit() has been called, False before.
        """
        return self._is_fitted

    def fit(self, _X: DataFrame, _y: Series | None = None) -> Self:
        """Learn parameters from training data.

        Returns:
            Self: The fitted preprocessor instance.
        """
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        """Apply transformation to data.

        Must be implemented by all subclasses.
        """


class StatefulPreprocessor(BasePreprocessor):
    """Base for transformers that delegate fit/transform to a ColumnTransformer.

    Subclasses must assign ``self._column_transformer`` in ``__init__``
    before any call to ``fit`` or ``transform``.
    """

    _column_transformer: ColumnTransformer

    def fit(self, X: DataFrame, y: Series | None = None) -> Self:
        """Fit the internal ColumnTransformer on training data.

        Args:
            X: Training DataFrame.
            y: Ignored; present for sklearn pipeline compatibility.

        Returns:
            Self: The fitted transformer instance.
        """
        self._column_transformer.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Apply the fitted ColumnTransformer to data.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame: Transformed output with canonical column order.
        """
        return cast("DataFrame", self._column_transformer.transform(X))
