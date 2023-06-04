import pytest
import requests

@pytest.fixture(scope="module")
def base_url():
    return "http://0.0.0.0:7777/seldon/default/intent-classification/api/v1.0/predictions"

def test_inference(base_url):
    data = { "data": { "ndarray": ["this is an example", "this is an example"] } }
    response = requests.post(base_url, json=data)
    assert response.status_code == 200
    expected_result = 2
    assert len(response.json()['data']["ndarray"][0]) == expected_result