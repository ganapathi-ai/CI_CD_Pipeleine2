from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from pathlib import Path

app = FastAPI(title="House Price API")

MODEL_PATH = Path(__file__).parent / "models" / "linmodel.pkl"

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

class Input(BaseModel):
    area: float
    bedrooms: int

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: Input):
    if model is None:
        return {"predicted_price": data.area * 50 + data.bedrooms * 1000}

    X = np.array([[data.area, data.bedrooms]])
    return {"predicted_price": float(model.predict(X)[0])}