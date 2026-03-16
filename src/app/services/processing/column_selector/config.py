from pydantic import BaseModel

SELECTED_FEATURES: list[str] = [
    "young_adult",
    "elderly",
    "waist_to_height_ratio",
    "told_high_cholesterol_missing",
    "told_high_bp",
    "told_high_cholesterol",
    "is_female",
    "drinking_frequency",
    "diastolic_bp",
    "systolic_bp",
    "education_level",
    "phq9_score",
    "vigorous_minutes_per_week",
]


class ColumnSelectorConfig(BaseModel):
    """Configuration for ColumnSelector transformer."""

    features: list[str] = SELECTED_FEATURES
