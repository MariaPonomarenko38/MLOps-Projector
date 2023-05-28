import pytest
import tritonclient.http as httpclient
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from client import get_features, get_predictions

@pytest.fixture(scope="module")
def triton_client():
    client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
    yield client
    client = None

@pytest.fixture(scope="module")
def extracted_features(triton_client):
    features = get_features(triton_client, "../Imgdirty_872_1.jpg")
    return features

def test_features_shape(extracted_features):
    expected_shape = (1, 7, 7, 1024)
    assert extracted_features.shape == expected_shape

def test_image_classification(triton_client, extracted_features):
    expected_prediction = 0
    predictions = get_predictions(triton_client, extracted_features)

    assert np.argmax(predictions) == expected_prediction