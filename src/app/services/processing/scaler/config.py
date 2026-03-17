from pydantic import BaseModel

BINARY_COLS: list[str] = [
    "young_adult",
    "elderly",
    "told_high_cholesterol_missing",
    "told_high_bp",
    "told_high_cholesterol",
    "is_female",
]

CONTINUOUS_COLS: list[str] = [
    "waist_to_height_ratio",
    "drinking_frequency",
    "diastolic_bp",
    "systolic_bp",
    "education_level",
    "phq9_score",
    "vigorous_minutes_per_week",
]


class ScalerConfig(BaseModel):
    """Configuration for Scaler transformer."""

    binary_cols: list[str] = BINARY_COLS
    continuous_cols: list[str] = CONTINUOUS_COLS
