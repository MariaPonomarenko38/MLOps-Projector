import pytest
from image_classification.data import load_data, get_features
from image_classification.utils import save_in_h5
import json 

@pytest.fixture(scope="session")
def args():
    with open('./data/config.json') as f:
        args = json.load(f)

    return args

@pytest.fixture(scope="session")
def args_path():
    path = './data/config.json'
    return path

@pytest.fixture(scope="session")
def raw_data(args):
    (train_images, train_labels) = load_data(args["path_to_data"] + 'Train', args["class_names"], args["image_size"])
    (test_images, test_labels) = load_data(args["path_to_data"] + 'Test', args["class_names"], args["image_size"])

    return train_images, train_labels, test_images, test_labels

@pytest.fixture(scope="session")
def processed_data(args, raw_data):
    train_images, train_labels, test_images, test_labels = raw_data
    train_features = get_features(args, train_images)
    test_features = get_features(args, test_images)

    save_in_h5(args["interim_path_data"] + 'train_labels.h5', train_labels)
    save_in_h5(args["interim_path_data"] + 'test_labels.h5', test_labels)
    save_in_h5(args["processed_path_data"] + 'train_features.h5', train_features)
    save_in_h5(args["processed_path_data"] + 'test_features.h5', test_features)
    return train_features, train_labels, test_features, test_labels
