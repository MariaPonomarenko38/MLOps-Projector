import logging
from pathlib import Path
from typing import List
import cv2
import wandb
import pickle 
from filelock import FileLock
from keras.applications import MobileNet
import numpy as np
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import json 
import tensorflow as tf
import os
from dotenv import load_dotenv
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from minio_client.minio_client import MinioClient

load_dotenv()

logger = logging.getLogger()

MODEL_ID = "mashaponomarenko2002/solar panels classification/model-stellar-bothan-49:v0"
MODEL_PATH = './models_loaded/'
MODEL_LOCK = ".lock-file"
MODEL_NAME = "saved_model.pb"
WANDB_PROJECT_NAME = "solar panels classification"


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init(project=WANDB_PROJECT_NAME) as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")

def open_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
        return data
    
def preprocessing(minio_client, bucket_name, object, image_size):
    image_data = minio_client.client.get_object(bucket_name, object.object_name).data
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return image

class Predictor:
    def __init__(self, model_load_path: str):
        self.model = tf.keras.models.load_model(model_load_path)
        self.mobile_net = MobileNet(input_shape=(150,150,3), weights='imagenet', include_top=False)

    def predict(self, bucket_name, folder):
    
        minio_client = MinioClient(os.getenv("MINIO_ACCESS_KEY"), os.getenv("MINIO_SECRET_KEY"),os.getenv("MINIO_URL"))
        objects = minio_client.get_data(bucket_name, folder)
        images = []
        for obj in objects:
            image = preprocessing(minio_client, bucket_name, obj, (150, 150))
            images.append(image)
        images = np.array(images, dtype = 'float32')
        images = images / 255.0
        features = self.mobile_net.predict(images)
        predictions = self.model.predict(features)
        pred_labels = np.argmax(predictions, axis = 1)
        return list(pred_labels)
    
    @classmethod
    def default_from_model_registry(cls) -> "Predictor":
        with FileLock(MODEL_LOCK):
            if not (Path(MODEL_PATH) / MODEL_NAME).exists():
                load_from_registry(model_name=MODEL_ID, model_path=MODEL_PATH)

        return cls(model_load_path=MODEL_PATH)