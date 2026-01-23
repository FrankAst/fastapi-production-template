from abc import abstractmethod
from typing import Self

from pandas import DataFrame, Series
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin


class AgeBinnerConfig(BaseModel):
    """Configuration for AgeBinner with sensible defaults."""

    age_column: str = "RIDAGEYR"
    age_bins: tuple[int, ...] = (18, 45, 65, 80)
    age_labels: tuple[str, ...] = ("young_adult", "middle_age", "senior")
    elderly_label: str = "elderly"
    unknown_label: str = "age_unknown"
    elderly_top_coded_age: int = 80
    age_group_col: str = "age_group"


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all preprocessing transformers."""

    def __init__(self, config: AgeBinnerConfig | None = None) -> None:
        self.config = config or AgeBinnerConfig()

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
