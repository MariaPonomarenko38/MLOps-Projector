import logging
from typing import List

from src.predictor import Predictor

logger = logging.getLogger()


class SeldonAPI:
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry()

    def predict(self, X, features_names: List[str]):
        logger.info(X)
        results = self.predictor.predict(X)
        logger.info(results)
        return results