import pandas as pd
from pandas import DataFrame, Series  # pyright: ignore[reportUnknownVariableType]

from app.services.processing.base import BasePreprocessor

from .config import AgeBinnerConfig


class AgeBinner(BasePreprocessor):
    """Bins age column into categories with one-hot encoding."""

    def __init__(self) -> None:
        super().__init__()
        self.config = AgeBinnerConfig()

    def _categorize_age(self, d: DataFrame) -> Series:
        """
        Categorize the age column into discrete age groups.

        Reads
        -----
        RIDAGEYR {self.config.age_column}: numeric
            Raw age in years.

        Produces
        --------
        age_group {self.config.age_group_col}: categorical
            Age category derived from configured bins
            (e.g. young_adult, middle_age, senior, elderly, unknown).

        Returns
        -------
        pandas.Series
            The computed age group labels for each row. This Series is later
            added to the DataFrame as the `age_group` column.
        """
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

    def _add_age_group_column(self, d: DataFrame) -> DataFrame:
        df = d.copy()
        df[self.config.age_group_col] = self._categorize_age(df)
        return df

    def _one_hot_encode(self, d: DataFrame) -> DataFrame:
        dummies = pd.get_dummies(d[self.config.age_group_col], dtype=int)
        dummies.columns = dummies.columns.astype(str)
        return d.join(dummies)

    def transform(self, X: DataFrame) -> DataFrame:
        return (
            X
            .pipe(self._add_age_group_column)
            .pipe(self._one_hot_encode)
            .drop(columns=[self.config.age_column, self.config.age_group_col])
        )
