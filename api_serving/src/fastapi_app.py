from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from src.predictor import Predictor


class Payload(BaseModel):
    text: str


class Prediction(BaseModel):
    intent: str


app = FastAPI()
predictor = Predictor.default_from_model_registry()


@app.get("/health_check")
def health_check() -> str:
    return "ok"


@app.post("/predict", response_model=Prediction)
def predict(payload: Payload) -> Prediction:
    prediction = predictor.predict([payload.text])[0]
    return Prediction(intent=prediction)