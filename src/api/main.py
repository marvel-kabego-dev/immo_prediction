# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(__file__))
from predict import predict_price

app = FastAPI(
    title="API Prédiction Immobilière",
    description="Estimation du prix d'un logement via XGBoost",
    version="1.0.0"
)

class HouseFeatures(BaseModel):
    GrLivArea: Optional[float] = 0
    OverallQual: Optional[float] = 0
    GarageCars: Optional[float] = 0
    TotalBsmtSF: Optional[float] = 0
    YearBuilt: Optional[float] = 0
    FullBath: Optional[float] = 0
    TotRmsAbvGrd: Optional[float] = 0
    Fireplaces: Optional[float] = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    prix = predict_price(features.model_dump())
    return {
        "prix_estime": f"{prix:,.0f}$",
        "prix_numerique": prix
    }