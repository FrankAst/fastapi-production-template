from pydantic import Field

from .base import BaseEntity


class ShapContribution(BaseEntity):
    feature: str = Field(
        description="Feature name from the post-processed feature space"
    )
    shap_value: float = Field(description="Log-odds contribution of the feature")
