from typing import cast

import numpy as np
from pandas import DataFrame, Series
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.domain import EvaluationTestSet, LRVifBicConfig, MLModel
from app.services.helper import load_artifact
from app.settings import Settings

from .evaluation_result import (
    ConfusionMatrixResult,
    EvaluationResult,
    MetricsResult,
    MetricWithCI,
)
from .exceptions import NoEvaluationArtifactsError


class EvaluationService(BaseModel):
    """Loads training artifacts and computes classification metrics
    with bootstrap CIs."""

    lr_config: LRVifBicConfig = Field(default_factory=LRVifBicConfig)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def evaluate(self) -> EvaluationResult:
        """Load held-out test set, predict at threshold, compute metrics and CIs.

        Propagates NoEvaluationArtifactsError if either artifact is missing on disk.

        Returns:
            EvaluationResult: Point estimates and bootstrap CIs for AP, precision,
                recall, F1, plus the full confusion matrix and evaluation metadata.
        """
        eval_model = self._load_eval_model()
        X_test, y_test = self._load_test_set()

        y_pred_proba: np.ndarray = eval_model.predict_proba(X_test)[:, 1]
        y_pred: np.ndarray = (y_pred_proba >= self.lr_config.threshold).astype(int)

        metrics = self._compute_metrics(y_test, y_pred, y_pred_proba)
        confusion = self._compute_confusion_matrix(y_test, y_pred)

        return EvaluationResult(
            threshold=self.lr_config.threshold,
            n_test=len(y_test),
            metrics=metrics,
            confusion_matrix=confusion,
        )

    @staticmethod
    def _load_eval_model() -> MLModel:
        artifact = load_artifact(Settings.EVAL_MODEL_PATH)
        if artifact is None:
            raise NoEvaluationArtifactsError
        return cast("MLModel", artifact)

    @staticmethod
    def _load_test_set() -> tuple[DataFrame, Series]:
        artifact = load_artifact(Settings.TEST_SET_PATH)
        if artifact is None:
            raise NoEvaluationArtifactsError
        bundle = cast("EvaluationTestSet", artifact)
        return bundle.X_test, bundle.y_test

    def _compute_metrics(
        self,
        y_test: Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> MetricsResult:
        """Compute point estimates and 95% bootstrap CIs for all four metrics.

        Uses Raschka Method 3: resample the predictions rather than refitting the model.
        A single bootstrap loop computes all four metrics per iteration to avoid
        redundant resampling.

        Args:
            y_test: True labels from the held-out test set.
            y_pred: Binary predictions at the configured threshold.
            y_pred_proba: Predicted probabilities for the positive class.

        Returns:
            MetricsResult with value, ci_lower, and ci_upper for each metric.
        """
        point_ap = float(average_precision_score(y_test, y_pred_proba))
        point_precision = float(precision_score(y_test, y_pred, zero_division=0))
        point_recall = float(recall_score(y_test, y_pred, zero_division=0))
        point_f1 = float(f1_score(y_test, y_pred, zero_division=0))

        y_arr = np.asarray(y_test)
        boot_ap, boot_pr, boot_re, boot_f1 = self._run_bootstrap(y_arr, y_pred_proba)

        return MetricsResult(
            average_precision=MetricWithCI(point_ap, *self._ci(boot_ap)),
            precision=MetricWithCI(point_precision, *self._ci(boot_pr)),
            recall=MetricWithCI(point_recall, *self._ci(boot_re)),
            f1=MetricWithCI(point_f1, *self._ci(boot_f1)),
        )

    def _run_bootstrap(
        self,
        y_arr: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Stratified resample predictions n_bootstrap_eval times and collect
        per-metric values.

        Resamples each class separately with replacement, preserving the original
        class counts per iteration. This guarantees both classes are always present
        in every bootstrap sample, making all four metrics well-defined without
        requiring zero_division guards.

        Args:
            y_arr: True labels as a numpy array.
            y_pred_proba: Predicted probabilities for the positive class.

        Returns:
            Four lists of bootstrap metric values: AP, precision, recall, F1.
        """
        rng = np.random.default_rng(self.lr_config.random_state)
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        boot_ap: list[float] = []
        boot_precision: list[float] = []
        boot_recall: list[float] = []
        boot_f1: list[float] = []

        for _ in range(self.lr_config.n_bootstrap_eval):
            idx = np.concatenate([
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ])
            y_boot, proba_boot = y_arr[idx], y_pred_proba[idx]
            pred_boot = (proba_boot >= self.lr_config.threshold).astype(int)

            boot_ap.append(float(average_precision_score(y_boot, proba_boot)))
            boot_precision.append(
                float(precision_score(y_boot, pred_boot, zero_division=0))
            )
            boot_recall.append(float(recall_score(y_boot, pred_boot, zero_division=0)))
            boot_f1.append(float(f1_score(y_boot, pred_boot, zero_division=0)))

        return boot_ap, boot_precision, boot_recall, boot_f1

    @staticmethod
    def _compute_confusion_matrix(
        y_test: Series, y_pred: np.ndarray
    ) -> ConfusionMatrixResult:
        cm = confusion_matrix(y_test, y_pred)
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        return ConfusionMatrixResult(tp=tp, fp=fp, tn=tn, fn=fn)

    @staticmethod
    def _ci(boots: list[float]) -> tuple[float, float]:
        arr = np.array(boots)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
