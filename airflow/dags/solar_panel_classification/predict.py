import pandas as pd
from alibi_detect.cd import MMDDrift
from alibi_detect.saving import save_detector, load_detector
import pickle
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator   
import tensorflow as tf    
import logging
from solar_panel_classification.utils import load_h5
from pathlib import Path
import numpy as np

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

def predict(inference_images_path: Path, model_load_path: Path, result_path: Path, 
            number_of_images_per_panel, **context):
    last_folder_name = context['task_instance'].xcom_pull(task_ids='load_data')
    is_drift = context['task_instance'].xcom_pull(task_ids='detect_drift')
    inference_images = load_h5(inference_images_path + f'{last_folder_name}_features.h5')
    predictor = Predictor(model_load_path)
    outputs = predictor.predict(inference_images)
    find_percentage_of_clear(outputs, number_of_images_per_panel, result_path)

def detect_drift(inference_images_path: Path, detector_path: Path, **context):
    last_folder_name = context['task_instance'].xcom_pull(task_ids='load_data')
    inference_images = load_h5(inference_images_path + f'{last_folder_name}_features.h5')
    detector = load_detector(detector_path)
    is_drift = detector.predict(inference_images)['data']['is_drift']
    return is_drift