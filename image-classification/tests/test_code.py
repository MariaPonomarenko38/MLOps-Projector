import numpy as np
import pytest
import sys
import os
from sklearn.metrics import f1_score, fbeta_score
from image_classification.utils import compute_metrics

@pytest.fixture()
def eval_pred():
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = np.array([1, 0, 1])

    _eval_pred = (predictions, labels)
    return _eval_pred

def test_compute_metrics(eval_pred):
    
    expected_result = {"f1": 1, "f0.5": 1, "accuracy": 1}
    
    result = compute_metrics(predictions=eval_pred[0], labels=eval_pred[1])

    assert result.keys() == expected_result.keys()
    for key in result.keys():
        assert abs(result[key] - expected_result[key]) < 1e-6