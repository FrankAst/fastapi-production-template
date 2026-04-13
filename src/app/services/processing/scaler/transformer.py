from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from app.services.processing.base import StatefulPreprocessor

from .config import BINARY_COLS, CONTINUOUS_COLS


class Scaler(StatefulPreprocessor):
    """Stateful scaler that standardises continuous features and passes binary
    features through unchanged.

    Strategy:
        - Continuous columns: standardised to mean=0, std=1 via StandardScaler.
        - Binary columns: passed through unscaled.
    """

    def __init__(self) -> None:
        super().__init__()
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), CONTINUOUS_COLS),
                ("passthrough", "passthrough", BINARY_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._column_transformer.set_output(transform="pandas")
