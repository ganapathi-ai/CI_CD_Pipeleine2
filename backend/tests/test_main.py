from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API running"}


def test_predict_valid_input():
    payload = {"area": 1200.5, "bedrooms": 3}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))