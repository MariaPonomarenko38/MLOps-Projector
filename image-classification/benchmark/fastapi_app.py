from fastapi import FastAPI

from predictor import Predictor
from pydantic import BaseModel
from typing import List

class Payload(BaseModel):
    bucket_name: str
    folder_name: str

class Prediction(BaseModel):
    labels: List[int]

app = FastAPI()
predictor = Predictor.default_from_model_registry()

@app.post("/predict", response_model=Prediction)

def predict(payload: Payload) -> Prediction:
    predictions = predictor.predict(payload.bucket_name, payload.folder_name)
    return Prediction(labels=predictions)