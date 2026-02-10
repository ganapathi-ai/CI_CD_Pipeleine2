import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_CI_CD.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

with open(MODEL_DIR / "linmodel.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully")