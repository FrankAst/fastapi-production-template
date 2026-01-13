import pandas as pd
from pandas import DataFrame, Series  # pyright: ignore[reportUnknownVariableType]

from . import preprocessingcfg as cfg
from .base_class import BasePreprocessor


class AgeBinner(BasePreprocessor):
    """Bins age column into categories with one-hot encoding.

    Age semantics:
    - 18-44  -> young_adult
    - 45-64  -> middle_age
    - 65-79  -> senior
    - 80+    -> elderly (top-coded as 80 in NHANES)
    - NaN    -> age_unknown
    """

    def __init__(self) -> None:
        self.age_group_col = "age_group"

    def transform(self, X: DataFrame) -> DataFrame:
        def categorize_age(d: DataFrame) -> Series:
            return (
                pd
                .cut(
                    d[cfg.AGE_COLUMN].where(
                        d[cfg.AGE_COLUMN] != cfg.ELDERLY_TOP_CODED_AGE
                    ),
                    bins=cfg.AGE_BINS,
                    labels=cfg.AGE_LABELS,
                    right=False,
                )
                .cat.add_categories([cfg.ELDERLY_LABEL, cfg.UNKNOWN_LABEL])
                .mask(d[cfg.AGE_COLUMN] == cfg.ELDERLY_TOP_CODED_AGE, cfg.ELDERLY_LABEL)
                .fillna(cfg.UNKNOWN_LABEL)
            )

        def one_hot_encode(d: DataFrame) -> DataFrame:
            dummies = pd.get_dummies(d[self.age_group_col]).astype(int)
            return pd.concat([d, dummies], axis=1)

        return (
            X
            .assign(**{self.age_group_col: categorize_age})
            .pipe(one_hot_encode)
            .drop(columns=[cfg.AGE_COLUMN, self.age_group_col])
        )
