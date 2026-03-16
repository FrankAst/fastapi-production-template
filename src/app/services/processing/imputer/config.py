from pydantic import BaseModel

ZERO_FILL_COLS: list[str] = ["vigorous_minutes_per_week", "drinking_frequency"]

MEDIAN_FILL_COLS: list[str] = [
    "waist_to_height_ratio",
    "told_high_bp",
    "told_high_cholesterol",
    "is_female",
    "diastolic_bp",
    "systolic_bp",
    "education_level",
    "phq9_score",
]

PASSTHROUGH_COLS: list[str] = [
    "young_adult",
    "elderly",
    "told_high_cholesterol_missing",
]


class ImputerConfig(BaseModel):
    """Configuration for Imputer transformer."""

    zero_fill_cols: list[str] = ZERO_FILL_COLS
    median_fill_cols: list[str] = MEDIAN_FILL_COLS
    passthrough_cols: list[str] = PASSTHROUGH_COLS
