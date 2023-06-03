from fastapi.testclient import TestClient
from src.fastapi_app import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.text.strip('"') == "ok"

def test_predict():
    payload = {"text": "How much a ticket to Paris costs?"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    assert response.json()["intent"] in ["flight", "flight_time", "airfare", "aircraft", "ground_service", 
                                         "airline", "bbreviation", "quantity"]