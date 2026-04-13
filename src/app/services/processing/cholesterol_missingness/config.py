from pydantic import BaseModel, ConfigDict, Field


class CholesterolMissingnessConfig(BaseModel):
    """Configuration for CholesterolMissingness transformer."""

    model_config = ConfigDict(frozen=True)

    source_column: str = Field(
        default="told_high_cholesterol",
        min_length=1,
        description="Column name for the source variable",
    )
    output_column: str = Field(
        default="told_high_cholesterol_missing",
        min_length=1,
        description="Column name for the output variable",
    )
