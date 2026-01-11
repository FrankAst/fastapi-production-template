import pandas as pd
from pandas import DataFrame, Series  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel

from . import preprocessingcfg as cfg


class ProcessingService(BaseModel):
    """Service for preprocessing data."""

    @staticmethod
    def age_binning(df: DataFrame) -> DataFrame:
        """
        Process age column into bins and applies one-hot encoding,
        explicitly handling missing values.
        Args:
            df (DataFrame): Input data.
        Returns:
            DataFrame: Data with binned age columns.

        Age semantics:
        - 18-44  -> young_adult
        - 45-64  -> middle_age
        - 65-79  -> senior
        - 80+    -> elderly (encoded as 80)
        - NaN    -> age_unknown
        """

        age_column = cfg.AGE_COLUMN
        labels = cfg.AGE_LABELS
        nan_label = cfg.UNKNOWN_LABEL
        elderly_label = cfg.ELDERLY_LABEL
        age_bins = cfg.AGE_BINS
        elderly_top_coded_age = cfg.ELDERLY_TOP_CODED_AGE
        age_group_col = "age_group"

        def categorize_age(d: DataFrame) -> Series:
            return (
                pd
                .cut(
                    d[age_column].where(d[age_column] != elderly_top_coded_age),
                    bins=age_bins,
                    labels=labels,
                    right=False,
                )
                .cat.add_categories([elderly_label, nan_label])
                .mask(d[age_column] == elderly_top_coded_age, elderly_label)
                .fillna(nan_label)
            )

        def one_hot_encode(d: DataFrame) -> DataFrame:
            dummies = pd.get_dummies(d[age_group_col]).astype(int)
            return pd.concat([d, dummies], axis=1)

        return (
            df
            .assign(age_group=categorize_age)
            .pipe(one_hot_encode)
            .drop(columns=[age_column, age_group_col])
        )

    @classmethod
    def preprocess(
        cls, df: DataFrame, *, skip: bool = False, training: bool = False
    ) -> DataFrame:
        """
        Preprocess the data.

        Args:
            df (DataFrame): Raw data.
            skip (bool): If True, skip preprocessing and return data as-is.
            training (bool): If True, indicates preprocessing for training data.
        Returns:
            DataFrame: Preprocessed data.
        """
        if skip:
            return df

        # Store the target column (last column) to preserve its position - IF TRAINING
        target_column = df.columns[-1] if training else None

        # Preprocessing logic
        steps = [
            cls.age_binning,
            # Other steps here
        ]

        for step in steps:
            df = step(df)

        # Ensure target column remains last
        if training and target_column is not None:
            columns = [col for col in df.columns if col != target_column] + [
                target_column
            ]
            return df[columns]
        return df
