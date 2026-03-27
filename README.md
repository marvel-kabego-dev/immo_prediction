# Prédiction du Prix Immobilier

Estimation du prix d'un bien immobilier par le biais d'un modèle de régression supervisée entraîné à partir de caractéristiques géographiques, structurelles et catégorielles.

## Stack technique
- Python 3.10+
- XGBoost, Scikit-learn, Pandas, NumPy
- FastAPI, Uvicorn

## Structure du projet
```
immo_prediction/
├── data/
│   ├── raw/          ← données brutes, immuables
│   └── processed/    ← données preprocessées
├── notebooks/        ← exploration et modélisation
├── src/api/          ← API REST
├── models/           ← modèles sauvegardés
└── outputs/          ← rapports et visualisations
```

## Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'API
```bash
cd src/api
uvicorn main:app --reload
```
API disponible sur `http://127.0.0.1:8000`
Documentation interactive sur `http://127.0.0.1:8000/docs`

## Exemple de prédiction
```json
POST /predict
{
  "GrLivArea": 1500,
  "OverallQual": 7,
  "GarageCars": 2,
  "YearBuilt": 2000
}
→ {"prix_estime": "298,400$"}
```

## Performances du modèle
| Métrique | Score |
|----------|-------|
| R²       | 0.905 |
| MAE      | ~14 500$ |
| RMSE     | 0.133 |