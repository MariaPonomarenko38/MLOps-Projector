import pandas as pd
from alibi_detect.cd import MMDDrift
from alibi_detect.saving import save_detector, load_detector
import pickle
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator   
import tensorflow as tf    
import logging
from utils import load_h5, get_args, setup_logger, compute_metrics 
from pathlib import Path
import numpy as np

class Predictor:
    def __init__(self, model_path: str, detector_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.detector = load_detector(detector_path)

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

def predict(test_images_path: Path, model_load_path: Path, detector_path: Path, 
            result_path: Path, number_of_images_per_panel):
    test_images = load_h5(test_images_path)
    predictor = Predictor(model_load_path, detector_path)
    is_drift = predictor.detector.predict(test_images)['data']['is_drift']
    if is_drift:
        print("Warning: Inference distribution differs significantly from training data.")
    outputs = predictor.predict(test_images)
    find_percentage_of_clear(outputs, number_of_images_per_panel, result_path)