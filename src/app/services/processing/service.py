import pandas as pd
from pandas import DataFrame  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel


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
        """

        age_column = "RIDAGEYR"
        bins = [18, 45, 65, 80]
        labels = ["young_adult", "middle_age", "elderly"]
        unknown_label = "age_unknown"

        # Initialize all output columns to 0
        for col in (*labels, unknown_label):
            df[col] = 0

        # Identify missing ages
        missing_mask = df[age_column].isna()

        # Explicit unknown bucket
        df.loc[missing_mask, unknown_label] = 1

        # Bin only non-missing values
        df.loc[~missing_mask, "age_group"] = pd.cut(
            df.loc[~missing_mask, age_column],
            bins=bins,
            labels=labels,
            right=False,
        )

        # One-hot encode known age groups
        age_dummies = df.loc[~missing_mask, "age_group"].str.get_dummies().astype(int)

        # Assign back
        for label in labels:
            if label in age_dummies.columns:
                df.loc[~missing_mask, label] = age_dummies[label]

        return df.drop(["age_group", age_column], axis=1)

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
