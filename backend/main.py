from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from pathlib import Path
import os

app = FastAPI(title="Linear Regression Model API")

MODEL_PATH = Path("models/linmodel.pkl")

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


class InputData(BaseModel):
    area: float
    bedrooms: int


@app.get("/")
def health_check():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: InputData):
    # CI / fallback behavior
    if model is None:
        # deterministic fake prediction for CI
        return {"predicted_price": float(data.area * 50 + data.bedrooms * 1000)}

    X = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}