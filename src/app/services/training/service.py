from pathlib import Path

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.domain import MLModel
from app.services.helper import load_model, save_model
from app.settings import Settings


class TrainingService(BaseModel):
    model_path: Path = Field(default=Settings.MODEL_PATH)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def model(self) -> MLModel:
        if self.model_path.exists():
            model = load_model(self.model_path)
            if model:
                return model

        return make_pipeline(StandardScaler(), LogisticRegression())  # type: ignore[return-value]

    def train(self, df: DataFrame) -> MLModel:
        """
        Train the ML model with provided features and target.
        Args:
            df (DataFrame): DataFrame where the last column is the target variable.
        Returns:
            MLModel: The trained machine learning model.
        """
        # Split features and target
        X = df.iloc[:, :-1]  # Features as DataFrame
        y = df.iloc[:, -1].tolist()  # Target as list of floats

        pipeline = self.model
        pipeline_fit = pipeline.fit(X, y)
        save_model(pipeline_fit, self.model_path)
        return pipeline
