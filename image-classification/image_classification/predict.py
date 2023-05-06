from pathlib import Path
import numpy as np
from keras.models import load_model
import h5py
from image_classification.utils import load_h5

class Predictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, data):
        predictions = self.model.predict(data)
        pred_labels = np.argmax(predictions, axis = 1)
        return pred_labels

def predict(test_images_path: Path, model_load_path: Path, result_path: Path):
    test_images = load_h5(test_images_path)
    predictor = Predictor(model_load_path)
    outputs = predictor.predict(test_images)
    np.save(result_path, outputs)
