import json
import csv
import uuid

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data/processed/abalone_clean.csv"

VERSION = "v3"
MODEL_PATH = BASE_DIR / f"models/model_{VERSION}.pkl"
METRICS_PATH = BASE_DIR / f"results/metrics_{VERSION}.json"
LOG_PATH = BASE_DIR / "results/experiments_log.csv"

SEED = 42

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["age", "rings"])
y = df["age"]

# Misma división que v1 y v2 para que las métricas sean comparables
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED
)

# El escalador forma parte del Pipeline: se ajusta solo con los datos de train
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gradient_boosting", GradientBoostingRegressor(random_state=SEED))
])

param_grid = {
    "gradient_boosting__n_estimators": [100, 200],
    "gradient_boosting__learning_rate": [0.05, 0.1],
    "gradient_boosting__max_depth": [2, 3, 4],
    "gradient_boosting__subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1,
    refit=True
)

# La búsqueda se realiza SOLAMENTE en train
grid.fit(X_train, y_train)

print("Mejores hiperparámetros:")
print(grid.best_params_)
print("MAE promedio en CV:", -grid.best_score_)

modelo = grid.best_estimator_

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE Test: {mae:.4f}")
print(f"RMSE Test: {rmse:.4f}")
print(f"R² Test: {r2:.4f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, MODEL_PATH, compress=3)

metricas = {
    "model_version": VERSION,
    "algorithm": "GradientBoostingRegressor",
    "scaler": "StandardScaler (dentro del Pipeline)",
    "random_state": SEED,
    "test_size": 0.2,
    "cv_folds": 3,
    "cv_scoring": "neg_mean_absolute_error",
    "best_cv_mae": float(-grid.best_score_),
    "best_hyperparameters": grid.best_params_,
    "test_metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=4)

experimento = {
    "experiment_id": str(uuid.uuid4()),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_version": VERSION,
    "hyperparameters": json.dumps(grid.best_params_, sort_keys=True),
    "random_state": SEED,
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

existe = LOG_PATH.exists()

with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=experimento.keys())

    if not existe or LOG_PATH.stat().st_size == 0:
        writer.writeheader()

    writer.writerow(experimento)
