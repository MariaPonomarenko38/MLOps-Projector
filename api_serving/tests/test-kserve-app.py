import pytest
import requests

@pytest.fixture(scope="module")
def base_url():
    return "http://localhost:8080/v1/models/intent-classification:predict"

def test_inference(base_url):
    data = { "instances": ["I need a ticket to Paris"] }
    response = requests.post(base_url, json=data)
    assert response.status_code == 200
    response_data = response.json()
    print(response_data)
    assert response_data["predictions"][0] == "flight"