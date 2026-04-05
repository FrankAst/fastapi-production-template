from dataclasses import dataclass


@dataclass
class MetricWithCI:
    """A single metric value with 95% bootstrap confidence interval bounds."""

    value: float
    ci_lower: float
    ci_upper: float


@dataclass
class MetricsResult:
    """Point estimates and CIs for all four classification metrics."""

    average_precision: MetricWithCI
    precision: MetricWithCI
    recall: MetricWithCI
    f1: MetricWithCI


@dataclass
class ConfusionMatrixResult:
    """Raw counts from the binary confusion matrix."""

    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class EvaluationResult:
    """Full evaluation output returned by EvaluationService.evaluate()."""

    threshold: float
    n_test: int
    metrics: MetricsResult
    confusion_matrix: ConfusionMatrixResult
