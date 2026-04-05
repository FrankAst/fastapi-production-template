from app.api.schema import BaseSchema


class MetricWithCISchema(BaseSchema):
    value: float
    ci_lower: float
    ci_upper: float


class MetricsSchema(BaseSchema):
    average_precision: MetricWithCISchema
    precision: MetricWithCISchema
    recall: MetricWithCISchema
    f1: MetricWithCISchema


class ConfusionMatrixSchema(BaseSchema):
    tp: int
    fp: int
    tn: int
    fn: int


class EvaluateResponse(BaseSchema):
    threshold: float
    n_test: int
    metrics: MetricsSchema
    confusion_matrix: ConfusionMatrixSchema
