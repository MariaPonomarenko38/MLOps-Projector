from pathlib import Path
import numpy as np
from keras.models import load_model
import h5py
import tensorflow as tf
import pandas as pd
from image_classification.utils import load_h5

class Predictor:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, data):
        predictions = self.model.predict(data)
        pred_labels = np.argmax(predictions, axis = 1)
        return pred_labels

def find_percentage_of_clear(array, m, result_path):
    num_groups = len(array) // m
    reshaped_array = array[:num_groups * m].reshape(num_groups, m)
    ones_count = np.sum(reshaped_array, axis=1)
    percentage_of_ones = (ones_count / m) * 100
    df = pd.DataFrame({'Number': range(1, num_groups+1),
                       'Percentage': percentage_of_ones})
    df.to_csv(result_path)

def predict(test_images_path: Path, model_load_path: Path, result_path: Path, number_of_images_per_panel, batch_size=None):
    test_images = load_h5(test_images_path)
    predictor = Predictor(model_load_path)
    outputs = predictor.predict(test_images[batch_size])
    find_percentage_of_clear(outputs, number_of_images_per_panel, result_path)