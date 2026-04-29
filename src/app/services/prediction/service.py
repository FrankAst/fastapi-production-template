from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import shap
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field

from app.domain import (
    BatchPredictionOutput,
    LRVifBicConfig,
    MLModel,
    PredictionInput,
    PredictionOutput,
    ShapContribution,
    ShapExplanation,
)
from app.services.helper import load_artifact, load_model
from app.services.processing.column_selector import SELECTED_FEATURES
from app.settings import Settings

from .exceptions import (
    NoBootstrapEnsembleError,
    NoShapBackgroundError,
    NoTrainedModelError,
)

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


class PredictionService(BaseModel):
    model_path: Path = Field(default_factory=lambda: Settings.PRODUCTION_MODEL_PATH)
    bootstrap_path: Path = Field(
        default_factory=lambda: Settings.BOOTSTRAP_ENSEMBLE_PATH
    )
    shap_background_path: Path = Field(
        default_factory=lambda: Settings.SHAP_BACKGROUND_PATH
    )
    lr_config: LRVifBicConfig = Field(default_factory=LRVifBicConfig)

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    @cached_property
    def production_model(self) -> MLModel:
        model = load_model(self.model_path)
        if model is None:
            raise NoTrainedModelError
        return model

    @cached_property
    def bootstrap_ensemble(self) -> list[MLModel]:
        ensemble = load_artifact(self.bootstrap_path)
        if ensemble is None:
            raise NoBootstrapEnsembleError
        return cast("list[MLModel]", ensemble)

    @cached_property
    def shap_background(self) -> DataFrame:
        background = load_artifact(self.shap_background_path)
        if background is None:
            raise NoShapBackgroundError
        return cast("DataFrame", background)

    @cached_property
    def explainer(self) -> shap.LinearExplainer:
        classifier = cast("Pipeline", self.production_model)[-1]
        return shap.LinearExplainer(classifier, self.shap_background)

    def predict(self, prediction_input: PredictionInput) -> PredictionOutput:
        """Score a single row with bootstrap CI bounds and a SHAP explanation.

        Args:
            prediction_input: Single-row feature DataFrame validated upstream.

        Returns:
            PredictionOutput with probability, 95% bootstrap CI bounds,
            screening flag, and the SHAP explanation in log-odds space.
        """
        df = prediction_input.features
        probability = float(self.production_model.predict_proba(df)[0, 1])
        ci_lower, ci_upper = self._bootstrap_ci(df)
        is_positive = probability > self.lr_config.threshold
        explanation = self._explain(df)

        return PredictionOutput(
            probability=probability,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_positive=is_positive,
            shap=explanation,
        )

    def batch_predict(self, prediction_input: PredictionInput) -> BatchPredictionOutput:
        """Legacy multi-row scoring used by the deferred ``/batch`` endpoint.

        Returns flat probabilities only, no CIs or SHAP. The future
        ``/batch`` redesign will replace this method.

        Args:
            prediction_input: Multi-row feature DataFrame validated upstream.

        Returns:
            BatchPredictionOutput with predictions and count.
        """
        raw_predictions = self.production_model.predict(prediction_input.features)
        predictions = [float(value) for value in raw_predictions]
        return BatchPredictionOutput(predictions=predictions, count=len(predictions))

    def _bootstrap_ci(self, df: DataFrame) -> tuple[float, float]:
        probas = np.array([
            model.predict_proba(df)[0, 1] for model in self.bootstrap_ensemble
        ])
        ci_lower, ci_upper = np.percentile(probas, [2.5, 97.5])
        return float(ci_lower), float(ci_upper)

    def _explain(self, df: DataFrame) -> ShapExplanation:
        preprocessor = cast("Pipeline", self.production_model)[:-1]
        transformed = preprocessor.transform(df)
        shap_values = np.asarray(self.explainer.shap_values(transformed))[0]
        base_value = float(np.asarray(self.explainer.expected_value).item())
        contributions = tuple(
            sorted(
                (
                    ShapContribution(feature=feature, shap_value=float(value))
                    for feature, value in zip(
                        SELECTED_FEATURES, shap_values, strict=True
                    )
                ),
                key=lambda contribution: abs(contribution.shap_value),
                reverse=True,
            )
        )
        final_value = base_value + sum(c.shap_value for c in contributions)
        return ShapExplanation(
            base_value=base_value,
            contributions=contributions,
            final_value=final_value,
        )
