import pytest

from app.domain import ShapContribution, ShapExplanation


def test_instantiates_with_base_contributions_and_final_value() -> None:
    base_value = -0.5
    final_value = -0.74
    contributions = (
        ShapContribution(feature="waist_to_height_ratio", shap_value=-0.42),
        ShapContribution(feature="systolic_bp", shap_value=0.18),
    )

    explanation = ShapExplanation(
        base_value=base_value,
        contributions=contributions,
        final_value=final_value,
    )

    assert explanation.base_value == pytest.approx(base_value)  # pyright: ignore[reportUnknownMemberType]
    assert explanation.final_value == pytest.approx(final_value)  # pyright: ignore[reportUnknownMemberType]
    assert explanation.contributions == contributions


def test_contributions_field_is_a_tuple_even_when_constructed_from_a_list() -> None:
    explanation = ShapExplanation.model_validate({
        "base_value": 0.0,
        "contributions": [
            {"feature": "phq9_score", "shap_value": 0.1},
            {"feature": "diastolic_bp", "shap_value": -0.05},
        ],
        "final_value": 0.05,
    })

    assert isinstance(explanation.contributions, tuple)
