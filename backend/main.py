from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="House Price API")

MODEL_PATH = Path(__file__).parent / "models" / "linmodel.pkl"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ci-cd-pipeleine2-1-frontend.onrender.com"],  # Or ["https://git-cicd-1-ngtw.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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