from pydantic import BaseModel


class CholesterolMissingnessConfig(BaseModel):
    """Configuration for CholesterolMissingness transformer."""

    source_column: str = "told_high_cholesterol"
    output_column: str = "told_high_cholesterol_missing"
