from collections.abc import Sequence

from fastapi import UploadFile
from pandas import DataFrame
from pydantic import ConfigDict, Field

from app.api.schema import BaseSchema
from app.domain import DrinkingFrequency, EducationLevel, SchemaValidator
from app.utils import process_csv_file


class SinglePredictionRequest(BaseSchema):
    RIDAGEYR: int = Field(alias="RIDAGEYR", ge=18, le=120, description="Age in years")
    BMXWAIST: float = Field(
        alias="BMXWAIST", ge=30, le=200, description="Waist circumference in cm"
    )
    BMXHT: float = Field(
        alias="BMXHT", ge=60, le=210, description="Standing height in cm"
    )
    told_high_bp: bool = Field(description="Ever told had high blood pressure")
    told_high_cholesterol: bool = Field(description="Ever told had high cholesterol")
    is_female: bool = Field(description="Sex flag (true if female)")
    drinking_frequency: DrinkingFrequency = Field(
        description="Drinking frequency category"
    )
    diastolic_bp: float = Field(ge=20, le=160, description="Diastolic blood pressure")
    systolic_bp: float = Field(ge=50, le=260, description="Systolic blood pressure")
    education_level: EducationLevel = Field(description="Education level category")
    phq9_score: int = Field(ge=0, le=27, description="PHQ-9 depression score 0-27")
    vigorous_minutes_per_week: int = Field(
        ge=0, le=2520, description="Vigorous activity minutes per week"
    )

    def to_validated_dataframe(self) -> DataFrame:
        """Build a single-row DataFrame validated against the Pandera schema.

        Returns:
            DataFrame: Single-row, schema-validated feature matrix ready for
            ``PredictionService.predict``.
        """
        df = DataFrame([self.model_dump(by_alias=False)])
        return SchemaValidator.validate_dataframe(df)


class BatchPredictionRequest(BaseSchema):
    file: UploadFile

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def from_upload(cls, file: UploadFile) -> DataFrame:
        """
        Create feature matrix from uploaded file.

        Validates the raw uploaded CSV against the training schema before
        any processing. This ensures the uploaded data has the correct
        structure, types, and columns expected by the model.

        Returns:
            DataFrame: Validated and processed feature data from the uploaded file.
        """
        df = await process_csv_file(file, require_target=False)

        return SchemaValidator.validate_dataframe(df)


class ShapContributionSchema(BaseSchema):
    feature: str = Field(
        description="Feature name from the post-processed feature space"
    )
    shap_value: float = Field(description="Log-odds contribution of the feature")


class ShapExplanationSchema(BaseSchema):
    base_value: float = Field(description="Explainer expected value in log-odds space")
    contributions: tuple[ShapContributionSchema, ...] = Field(
        description="Per-feature contributions, sorted by abs(shap_value) descending"
    )
    final_value: float = Field(
        description="base_value + sum(shap_value); equals the model's log-odds output"
    )


class SinglePredictionResponse(BaseSchema):
    probability: float = Field(
        description="Predicted probability of the positive class"
    )
    ci_lower: float = Field(description="Lower bound of the 95% bootstrap CI")
    ci_upper: float = Field(description="Upper bound of the 95% bootstrap CI")
    is_positive: bool = Field(
        description="True when probability exceeds the screening threshold"
    )
    shap: ShapExplanationSchema = Field(
        description="SHAP explanation of the prediction"
    )


class BatchPredictionResponse(BaseSchema):
    predictions: Sequence[float] = Field(description="Array of prediction results")
    count: int = Field(description="Number of predictions made")
