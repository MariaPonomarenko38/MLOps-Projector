import logging
from pathlib import Path
from typing import List

import wandb
import pickle 
from filelock import FileLock
import numpy as np
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model

logger = logging.getLogger()

MODEL_ID = "mashaponomarenko2002/intent-classification/intent-classification-model:v0"
MODEL_PATH = "./temp/model/"
MODEL_LOCK = ".lock-file"
MODEL_NAME = "intent_classification.h5"
WANDB_PROJECT_NAME = "intent-classification"


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init(project=WANDB_PROJECT_NAME) as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


class Predictor:
    def __init__(self, model_load_path: str, model_name: str):
        with open(model_load_path + 'tokenizer.pickle', 'rb') as handle:
            self.tokenizer  = pickle.load(handle)
        self.model = load_model(model_load_path + model_name)

    
    def predict(self, text: List[str]):
        preprocessed_text = np.array([message.replace('\d+', '') for message in text])
        preprocessed_text = self.tokenizer.texts_to_sequences(preprocessed_text)
        preprocessed_text = pad_sequences(preprocessed_text, 25)
        predictions = np.argmax(self.model .predict(preprocessed_text), axis=1)
        return predictions
    
    @classmethod
    def default_from_model_registry(cls) -> "Predictor":
        with FileLock(MODEL_LOCK):
            if not (Path(MODEL_PATH) / MODEL_NAME).exists():
                load_from_registry(model_name=MODEL_ID, model_path=MODEL_PATH)

        return cls(model_load_path=MODEL_PATH, model_name=MODEL_NAME)