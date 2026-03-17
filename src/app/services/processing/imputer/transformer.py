from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from app.services.processing.base import StatefulPreprocessor

from .config import ImputerConfig


class Imputer(StatefulPreprocessor):
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
