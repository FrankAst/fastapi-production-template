from pydantic import Field

from .base import BaseEntity
from .shap_explanation import ShapExplanation


class PredictionOutput(BaseEntity):
    probability: float = Field(
        description="Predicted probability of the positive class"
    )
    ci_lower: float = Field(description="Lower bound of the 95% bootstrap CI")
    ci_upper: float = Field(description="Upper bound of the 95% bootstrap CI")
    is_positive: bool = Field(
        description="True when probability exceeds the screening threshold"
    )
    shap: ShapExplanation = Field(description="SHAP explanation of the prediction")
