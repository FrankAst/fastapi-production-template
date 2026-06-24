from fastapi.openapi.models import Example

EXAMPLES: dict[str, Example] = {
    "normal": {
        "summary": "Realistic adult patient",
        "description": "Clinically plausible record that should produce a prediction.",
        "value": {
            "RIDAGEYR": 55,
            "BMXWAIST": 95.0,
            "BMXHT": 170.0,
            "toldHighBp": False,
            "toldHighCholesterol": False,
            "isFemale": True,
            "drinkingFrequency": 1,
            "diastolicBp": 80.0,
            "systolicBp": 130.0,
            "educationLevel": 3,
            "phq9Score": 4,
            "vigorousMinutesPerWeek": 90,
        },
    },
}
