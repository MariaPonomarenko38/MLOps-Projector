from minio_deployment.src.minio_client import MinioClient
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from alibi_detect.cd import KSDrift
from alibi_detect.saving import save_detector, load_detector
import pickle
import wandb
from pathlib import Path
import numpy as np

class Predictor:
    def __init__(self, model_path: str, detector_path: str):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        self.airline_num_drift_detector = load_detector(detector_path)

    def predict(self, dataframe, threshold=0.4):
        predictions = self.model.predict(dataframe)
        drift_predictions = self.airline_num_drift_detector.predict(np.array([dataframe["Airline_num"]]), 
                                                           return_p_val=True, return_distance=True)
        if drift_predictions['data']['p_val'] < threshold:
            print("Warning: Inference distribution differs significantly from training data.")
        return predictions
