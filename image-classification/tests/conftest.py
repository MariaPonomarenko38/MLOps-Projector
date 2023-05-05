import pytest
from image_classification.data import load_data
import json 
import numpy as np


@pytest.fixture(scope="session")
def args():
    with open('./data/config.json') as f:
        args = json.load(f)

    return args

@pytest.fixture(scope="session")
def data():
    with open('./data/config.json') as f:
        args = json.load(f)
    (train_images, train_labels) = load_data(args["path_to_data"] + 'Train', args["class_names"], args["image_size"])
    (test_images, test_labels) = load_data(args["path_to_data"] + 'Test', args["class_names"], args["image_size"])

    return train_images, train_labels, test_images, test_labels

@pytest.fixture(scope="session")
def data_testing():
    train_features = np.load('./test_data/test_processed_data/train_features.npy')
    test_features = np.load('./test_data/test_processed_data/test_features.npy')
    train_labels =  np.load('./test_data/test_interim_data/train_labels.npy')
    test_labels =  np.load('./test_data/test_interim_data/test_labels.npy')

    return train_features, train_labels, test_features, test_labels
