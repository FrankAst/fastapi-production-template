from pathlib import Path

from pydantic import BaseModel, Field

from app.domain import BatchPredictionOutput, MLModel, PredictionInput
from app.services.helper import load_model
from app.settings import Settings

from .exceptions import NoTrainedModelError


class PredictionService(BaseModel):
    model_path: Path = Field(default=Settings.PRODUCTION_MODEL_PATH)

    @property
    def model(self) -> MLModel | None:
        return load_model(self.model_path)

    def predict(self, prediction_input: PredictionInput) -> BatchPredictionOutput:
        """
        Make predictions for a single or batch input.

        Args:
            prediction_input: PredictionInput instance containing
            features for prediction.

        Returns:
            BatchPredictionOutput: Contains predictions and count.

        Raises:
            NoTrainedModelError: If no trained model is found.
        """
        if self.model is None:
            raise NoTrainedModelError

        prediction_results = self.model.predict(prediction_input.features)

        return BatchPredictionOutput(
            predictions=prediction_results, count=len(prediction_results)
        )
