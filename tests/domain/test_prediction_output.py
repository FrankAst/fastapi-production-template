import pytest

from app.domain import (
    PredictionOutput,
    ShapContribution,
    ShapExplanation,
)


def test_instantiates_with_probability_ci_flag_and_shap() -> None:
    probability = 0.42
    ci_lower = 0.35
    ci_upper = 0.49
    explanation = ShapExplanation(
        base_value=-0.5,
        contributions=(ShapContribution(feature="systolic_bp", shap_value=0.18),),
        final_value=-0.32,
    )

    output = PredictionOutput(
        probability=probability,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_positive=False,
        shap=explanation,
    )

    assert output.probability == pytest.approx(probability)  # pyright: ignore[reportUnknownMemberType]
    assert output.ci_lower == pytest.approx(ci_lower)  # pyright: ignore[reportUnknownMemberType]
    assert output.ci_upper == pytest.approx(ci_upper)  # pyright: ignore[reportUnknownMemberType]
    assert output.is_positive is False
    assert output.shap is explanation
