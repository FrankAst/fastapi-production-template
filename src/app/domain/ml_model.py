from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

import numpy as np
from pandas import DataFrame


@runtime_checkable
class MLModel(Protocol):
    def predict(self, X: DataFrame) -> Sequence[float]: ...

    def predict_proba(self, X: DataFrame) -> np.ndarray: ...

    def fit(self, X: DataFrame, y: Sequence[float]) -> Self: ...
