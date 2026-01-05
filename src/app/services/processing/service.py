from pandas import DataFrame, cut  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel


class ProcessingService(BaseModel):
    """Service for preprocessing data."""

    @staticmethod
    def age_binning(df: DataFrame) -> DataFrame:
        """
        Process age column into bins and applies
        one-hot encoding.
        Args:
            df (DataFrame): Input data.
        Returns:
            DataFrame: Data with binned age columns.
        """
        age_column = "RIDAGEYR"
        bins = [18, 45, 65, 80]
        labels = ["young_adult", "middle_age", "elderly"]

        # Create age group categories
        df["age_group"] = cut(df[age_column], bins=bins, labels=labels, right=False)

        # Create binary columns for each age group using one-hot encoding
        age_dummies = df["age_group"].str.get_dummies()

        # Add binary columns to dataframe
        for label in labels:
            if label in age_dummies.columns:
                df[label] = age_dummies[label].astype(int)
            else:
                df[label] = 0

        # Drop the intermediate age_group column and original age column
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
