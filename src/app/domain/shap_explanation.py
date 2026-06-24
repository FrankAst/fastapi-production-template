from pydantic import Field

from .base import BaseEntity
from .shap_contribution import ShapContribution


class ShapExplanation(BaseEntity):
    base_value: float = Field(description="Explainer expected value in log-odds space")
    contributions: tuple[ShapContribution, ...] = Field(
        description="Per-feature contributions, sorted by abs(shap_value) descending"
    )
    final_value: float = Field(
        description="base_value + sum(shap_value); equals the model's log-odds output"
    )
