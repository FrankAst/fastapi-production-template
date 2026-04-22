from pydantic import BaseModel, ConfigDict, Field


class AgeBinnerConfig(BaseModel):
    """Configuration for AgeBinner with sensible defaults."""

    model_config = ConfigDict(frozen=True)

    age_column: str = Field(
        default="RIDAGEYR",
        min_length=1,
        description="Column name for age in the dataset",
    )
    age_bins: tuple[int, ...] = (18, 45, 65, 80)
    age_labels: tuple[str, ...] = ("young_adult", "middle_age", "senior")
    elderly_label: str = Field(
        default="elderly", min_length=1, description="Label for the elderly age group"
    )
    unknown_label: str = Field(
        default="age_unknown", min_length=1, description="Label for unknown age values"
    )
    elderly_top_coded_age: int = 80
    age_group_col: str = Field(
        default="age_group",
        min_length=1,
        description="Column name for the age group variable",
    )
