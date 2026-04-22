from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from app.services.processing.base import StatefulPreprocessor

from .config import MEDIAN_FILL_COLS, PASSTHROUGH_COLS, ZERO_FILL_COLS


class Imputer(StatefulPreprocessor):
    """Stateful imputer applying per-column-group strategies via ColumnTransformer.

    Strategy:
        - Zero-fill: columns where NaN means "doesn't do this activity"
        - Median-fill: columns with structural or random missingness
        - Passthrough: engineered binary flags that are already complete
    """

    def __init__(self) -> None:
        super().__init__()
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "zero_fill",
                    SimpleImputer(strategy="constant", fill_value=0),
                    ZERO_FILL_COLS,
                ),
                (
                    "median_fill",
                    SimpleImputer(strategy="median"),
                    MEDIAN_FILL_COLS,
                ),
                ("passthrough", "passthrough", PASSTHROUGH_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._column_transformer.set_output(transform="pandas")
