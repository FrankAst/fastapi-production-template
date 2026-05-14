from collections.abc import Sequence

from fastapi import UploadFile
from pandas import DataFrame
from pydantic import Field

from app.api.schema import BaseSchema
from app.domain import DrinkingFrequency, EducationLevel, SchemaValidator
from app.utils import process_csv_file


class SinglePredictionRequest(BaseSchema):
    RIDAGEYR: int = Field(alias="RIDAGEYR", ge=18, le=120, description="Age in years")
    BMXWAIST: float | None = Field(
        default=None,
        alias="BMXWAIST",
        ge=30,
        le=200,
        description="Waist circumference in cm",
    )
    BMXHT: float | None = Field(
        default=None,
        alias="BMXHT",
        ge=60,
        le=210,
        description="Standing height in cm",
    )
    told_high_bp: bool | None = Field(
        default=None, description="Ever told had high blood pressure"
    )
    told_high_cholesterol: bool | None = Field(
        default=None, description="Ever told had high cholesterol"
    )
    is_female: bool = Field(description="Sex flag (true if female)")
    drinking_frequency: DrinkingFrequency | None = Field(
        default=None, description="Drinking frequency category"
    )
    diastolic_bp: float | None = Field(
        default=None, ge=20, le=160, description="Diastolic blood pressure"
    )
    systolic_bp: float | None = Field(
        default=None, ge=50, le=260, description="Systolic blood pressure"
    )
    education_level: EducationLevel = Field(description="Education level category")
    phq9_score: int | None = Field(
        default=None, ge=0, le=27, description="PHQ-9 depression score 0-27"
    )
    vigorous_minutes_per_week: int | None = Field(
        default=None,
        ge=0,
        le=2520,
        description="Vigorous activity minutes per week",
    )

    def to_validated_dataframe(self) -> DataFrame:
        """Build a single-row DataFrame validated against the Pandera schema.

        Returns:
            DataFrame: Single-row, schema-validated feature matrix ready for
            ``PredictionService.predict``.
        """
        df = DataFrame([self.model_dump(by_alias=False)])
        return SchemaValidator.validate_dataframe(df)


async def parse_prediction_upload(file: UploadFile) -> DataFrame:
    """Parse and validate an uploaded CSV for batch prediction.

    Accepts a target-free CSV (labels are not provided at inference time) and
    validates it against the static prediction schema.

    Returns:
        DataFrame: Validated and coerced feature matrix.
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
