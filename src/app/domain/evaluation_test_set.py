from dataclasses import dataclass

from pandas import DataFrame, Series


@dataclass
class EvaluationTestSet:
    """Held-out test set persisted by TrainingService and loaded by
    EvaluationService."""

    X_test: DataFrame
    y_test: Series
