# src/api/predict.py
import joblib
import numpy as np
import pandas as pd

model = joblib.load('../../models/xgb_champion.pkl')
expected_columns = joblib.load('../../models/expected_columns.pkl')

def predict_price(features: dict) -> float:
    df = pd.DataFrame([features])
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    prix_log = model.predict(df)[0]
    prix_dollars = float(np.exp(prix_log))
    
    return round(prix_dollars, 2)