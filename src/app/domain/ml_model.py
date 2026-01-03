from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

from pandas import DataFrame


@runtime_checkable
class MLModel(Protocol):
    def predict(self, X: DataFrame) -> Sequence[float]: ...

    def fit(self, X: DataFrame, y: Sequence[float]) -> Self: ...
