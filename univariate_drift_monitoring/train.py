from minio_deployment.src.minio_client import MinioClient
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from alibi_detect.cd import KSDrift
from alibi_detect.saving import save_detector
import pickle
import wandb
from pathlib import Path

def train(data_path, model_path, detector_path):
    data = pd.read_csv(data_path)
    y = data['Delay']
    X = data.drop('Delay', axis=1)
    model = DecisionTreeClassifier(criterion='entropy',max_depth=16,random_state=40)
    model.fit(X, y)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    ref = X["Airline_num"].to_numpy()
    airline_num_drift_detector = KSDrift(ref, p_val=0.01)
    save_detector(airline_num_drift_detector, detector_path)