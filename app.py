import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Chargement unique au démarrage
@st.cache_resource
def load_model():
    model = joblib.load('models/xgb_champion.pkl')
    columns = joblib.load('models/expected_columns.pkl')
    return model, columns

model, expected_columns = load_model()

# Titre
st.title("🏠 Estimateur de Prix Immobilier")
st.write("Entrez les caractéristiques du logement pour obtenir une estimation.")

# Inputs utilisateur
col1, col2 = st.columns(2)

with col1:
    GrLivArea = st.number_input("Surface habitable (m²)", min_value=0, value=1500)
    OverallQual = st.slider("Qualité globale (1-10)", 1, 10, 7)
    YearBuilt = st.number_input("Année de construction", min_value=1800, max_value=2024, value=2000)
    FullBath = st.number_input("Salles de bain complètes", min_value=0, max_value=5, value=2)

with col2:
    GarageCars = st.number_input("Capacité garage (voitures)", min_value=0, max_value=5, value=2)
    TotalBsmtSF = st.number_input("Surface sous-sol (m²)", min_value=0, value=800)
    TotRmsAbvGrd = st.number_input("Nombre de pièces", min_value=0, max_value=20, value=6)
    Fireplaces = st.number_input("Nombre de cheminées", min_value=0, max_value=5, value=1)

# Bouton de prédiction
if st.button("Estimer le prix 🏷️"):
    features = {
        'GrLivArea': GrLivArea,
        'OverallQual': OverallQual,
        'YearBuilt': YearBuilt,
        'FullBath': FullBath,
        'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF,
        'TotRmsAbvGrd': TotRmsAbvGrd,
        'Fireplaces': Fireplaces
    }
    
    df = pd.DataFrame([features])
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    prix_log = model.predict(df)[0]
    prix = float(np.exp(prix_log))
    
    st.success(f"### Prix estimé : {prix:,.0f} $")
    st.caption("Estimation basée sur un modèle XGBoost entraîné sur des données immobilières américaines.")