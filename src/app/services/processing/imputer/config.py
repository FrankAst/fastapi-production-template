ZERO_FILL_COLS: tuple[str, ...] = ("vigorous_minutes_per_week", "drinking_frequency")

MEDIAN_FILL_COLS: tuple[str, ...] = (
    "waist_to_height_ratio",
    "told_high_bp",
    "told_high_cholesterol",
    "is_female",
    "diastolic_bp",
    "systolic_bp",
    "education_level",
    "phq9_score",
)

PASSTHROUGH_COLS: tuple[str, ...] = (
    "young_adult",
    "elderly",
    "told_high_cholesterol_missing",
)
