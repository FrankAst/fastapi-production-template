from pydantic import BaseModel


class WaistToHeightRatioConfig(BaseModel):
    """Configuration for WaistToHeightRatio transformer."""

    waist_column: str = "BMXWAIST"
    height_column: str = "BMXHT"
    output_column: str = "waist_to_height_ratio"
