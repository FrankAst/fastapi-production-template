import pytest
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture
def realistic_payload() -> dict[str, object]:
    return {
        "RIDAGEYR": 45,
        "BMXWAIST": 95.0,
        "BMXHT": 170.0,
        "toldHighBp": True,
        "toldHighCholesterol": False,
        "isFemale": True,
        "drinkingFrequency": 2,
        "diastolicBp": 80.0,
        "systolicBp": 120.0,
        "educationLevel": 4,
        "phq9Score": 5,
        "vigorousMinutesPerWeek": 30,
    }
