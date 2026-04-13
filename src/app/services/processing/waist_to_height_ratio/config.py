from pydantic import BaseModel, Field


class WaistToHeightRatioConfig(BaseModel):
    """Configuration for WaistToHeightRatio transformer."""

    waist_column: str = Field(
        default="BMXWAIST",
        min_length=1,
        description="The name of the column containing waist measurements.",
    )
    height_column: str = Field(
        default="BMXHT",
        min_length=1,
        description="The name of the column containing height measurements.",
    )
    output_column: str = Field(
        default="waist_to_height_ratio",
        min_length=1,
        description="The name of the column to store the waist-to-height ratio.",
    )
