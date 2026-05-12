from pydantic import Field

from app.api.schema import BaseSchema


class MetricWithCISchema(BaseSchema):
    value: float = Field(ge=0.0, le=1.0)
    ci_lower: float = Field(ge=0.0, le=1.0)
    ci_upper: float = Field(ge=0.0, le=1.0)


class MetricsSchema(BaseSchema):
    average_precision: MetricWithCISchema
    precision: MetricWithCISchema
    recall: MetricWithCISchema
    f1: MetricWithCISchema


class ConfusionMatrixSchema(BaseSchema):
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    tn: int = Field(ge=0)
    fn: int = Field(ge=0)


class EvaluateResponse(BaseSchema):
    threshold: float = Field(gt=0.0, lt=1.0)
    n_test: int = Field(gt=0)
    metrics: MetricsSchema
    confusion_matrix: ConfusionMatrixSchema
