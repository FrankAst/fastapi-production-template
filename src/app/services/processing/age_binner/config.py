from pydantic import BaseModel


class AgeBinnerConfig(BaseModel):
    """Configuration for AgeBinner with sensible defaults."""

    age_column: str = "RIDAGEYR"
    age_bins: tuple[int, ...] = (18, 45, 65, 80)
    age_labels: tuple[str, ...] = ("young_adult", "middle_age", "senior")
    elderly_label: str = "elderly"
    unknown_label: str = "age_unknown"
    elderly_top_coded_age: int = 80
    age_group_col: str = "age_group"
