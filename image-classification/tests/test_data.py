import numpy as np

def test_data_shape(raw_data):
    train_images, train_labels, test_images, test_labels = raw_data
    assert train_images.shape[0] + test_images.shape[0] == 9
    assert train_images.shape[1] == train_images.shape[1] == 224
    assert train_labels.shape[0] + test_labels.shape[0] == 9

def test_data_pixel_range(raw_data):
    train_images, _, test_images, _ = raw_data
    for image in train_images:
        assert np.all(np.logical_and(image >= 0, image <= 1))
    for image in test_images:
        assert np.all(np.logical_and(image >= 0, image <= 1))
