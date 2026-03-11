from abc import abstractmethod
from typing import Self

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all preprocessing transformers."""

    def __init__(self) -> None:
        pass

    def fit(self, _X: DataFrame, _y: Series | None = None) -> Self:
        """Learn parameters from training data.
        Returns:
            Self: The fitted preprocessor instance.
        """
        return self

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        """Apply transformation to data.

        Must be implemented by all subclasses.
        """
