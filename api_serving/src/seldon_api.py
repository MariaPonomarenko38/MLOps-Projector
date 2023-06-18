import logging
from typing import List
import time
from collections import defaultdict
from src.predictor import Predictor

logger = logging.getLogger()

class Score:
    def __init__(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.precision = defaultdict(int)
        self.recall = defaultdict(int)
        self.f1 = defaultdict(int)
        self.accuracy = 0

    def calculate_scores(self, y_true, y_pred):
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                self.tp[true] += 1
            else:
                self.fp[pred] += 1
                self.fn[true] += 1
        
        for class_name in self.tp.keys():
            self.precision[class_name] = 0
            self.recall[class_name] = 0
            self.f1[class_name] = 0
            if (self.tp[class_name] + self.fp[class_name]) != 0:
                self.precision[class_name] = self.tp[class_name] / (self.tp[class_name] + self.fp[class_name]) 
            if (self.tp[class_name] + self.fp[class_name]) != 0:
                self.recall[class_name] = self.tp[class_name] / (self.tp[class_name] + self.fn[class_name]) 
            if (self.precision[class_name] + self.recall[class_name]) != 0:
                self.f1[class_name] = 2 * (self.precision[class_name] * self.recall[class_name]) / (self.precision[class_name] + self.recall[class_name])
        
        tp_sum = sum(self.tp.values()) 
        self.accuracy = tp_sum / (tp_sum + sum(self.fp.values()))

class SeldonAPI:
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry()
        self.predict_time = None
        self.score = Score()

    def predict(self, X, features_names: List[str]):
        logger.info(X)
        start_time = time.time()
        results = self.predictor.predict(X)
        self.predict_time = time.time() - start_time
        logger.info(results)
        return results
    
    def metrics(self):
        macro_precision = sum(self.score.precision.values()) / len(self.predictor.class_names)
        macro_recall = sum(self.score.recall.values()) / len(self.predictor.class_names)
        macro_f1 = sum(self.score.f1.values()) / len(self.predictor.class_names)

        return [ 
            {"type": "GAUGE", "key": "gauge_runtime", "value": self.predict_time},
            {"type": "GAUGE", "key": f"accuracy", "value": self.score.accuracy},
            {"type": "GAUGE", "key": f"macro_averaged_precision", "value": macro_precision},
            {"type": "GAUGE", "key": f"macro_averaged_recall", "value": macro_recall},
            {"type": "GAUGE", "key": f"macro_averaged_f1_score", "value": macro_f1}]

    
    def send_feedback(self, features, feature_names, reward, truth, routing=""):
        logger.info("features")
        logger.info(features)

        logger.info("truth")
        logger.info(truth)

        results = self.predict(features, feature_names)
    
        self.score.calculate_scores(truth[0], results[0])

        return []