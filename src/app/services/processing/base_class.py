# src/app/services/processing/base.py

from abc import abstractmethod

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all preprocessing transformers."""

    def fit(self, _X: DataFrame, _y: Series | None = None) -> "BasePreprocessor":
        """Learn parameters from training data.

        Override this method for stateful transformers.
        Default implementation does nothing (stateless)

        Returns:
            BasePreprocessor: The fitted preprocessor instance.
        """
        return self

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        """Apply transformation to data.

        Must be implemented by all subclasses.
        """
