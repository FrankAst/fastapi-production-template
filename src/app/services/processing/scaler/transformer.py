from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from app.services.processing.base import StatefulPreprocessor

from .config import ScalerConfig


class Scaler(StatefulPreprocessor):
    """Stateful scaler that standardises continuous features and passes binary
    features through unchanged.

    Strategy:
        - Continuous columns: standardised to mean=0, std=1 via StandardScaler.
        - Binary columns: passed through unscaled.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = ScalerConfig()
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), self.config.continuous_cols),
                ("passthrough", "passthrough", self.config.binary_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._column_transformer.set_output(transform="pandas")
