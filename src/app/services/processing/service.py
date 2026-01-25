import pandas as pd
from pandas import DataFrame, Series  # pyright: ignore[reportUnknownVariableType]

from .base import BasePreprocessor


class AgeBinner(BasePreprocessor):
    """Bins age column into categories with one-hot encoding."""

    def __init__(self) -> None:
        super().__init__()
        self.age_group_col = "age_group"
        self.age_group_column = {self.config.age_group_col: self._categorize_age}

    def _categorize_age(self, d: DataFrame) -> Series:
        return (
            pd
            .cut(
                d[self.config.age_column].where(
                    d[self.config.age_column] != self.config.elderly_top_coded_age
                ),
                bins=self.config.age_bins,
                labels=self.config.age_labels,
                right=False,
            )
            .cat.add_categories([self.config.elderly_label, self.config.unknown_label])
            .mask(
                d[self.config.age_column] == self.config.elderly_top_coded_age,
                self.config.elderly_label,
            )
            .fillna(self.config.unknown_label)
        )

    def _one_hot_encode(self, d: DataFrame) -> DataFrame:
        dummies = pd.get_dummies(d[self.config.age_group_col]).astype(int)
        return pd.concat([d, dummies], axis=1)

    def transform(self, X: DataFrame) -> DataFrame:

        return (
            X
            .assign(**self.age_group_column)
            .pipe(self._one_hot_encode)
            .drop(columns=[self.config.age_column, self.config.age_group_col])
        )
