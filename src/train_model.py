import json
import csv
import uuid

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data/processed/abalone_clean.csv"

MODEL_PATH = BASE_DIR / "models/model_v1.pkl"
METRICS_PATH = BASE_DIR / "results/metrics_v1.json"
LOG_PATH = BASE_DIR / "results/experiments_log.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["age", "rings"])
y = df["age"]

SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED
)

pipe = Pipeline([
    ("random_forest", RandomForestRegressor(
        random_state=SEED,
        n_jobs=1
    ))
])

# Espacio de búsqueda
param_grid = {
    "random_forest__n_estimators": [80, 150],
    "random_forest__max_depth": [12, 20, None],
    "random_forest__min_samples_leaf": [1, 3]
}

# Búsqueda con validación cruzada
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

# Mejor configuración
print("Mejores hiperparámetros:")
print(grid.best_params_)

print("MAE promedio en CV:", -grid.best_score_)

# El mejor modelo ya fue reentrenado con todo train
modelo = grid.best_estimator_

# Evaluación final en test
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE Test: {mae:.4f}")
print(f"RMSE Test: {rmse:.4f}")
print(f"R² Test: {r2:.4f}")



MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    modelo,
    MODEL_PATH,
    compress=3
)

metricas = {
    "model_version": "v1",
    "algorithm": "RandomForestRegressor",
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

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=4)


experimento = {
    "experiment_id": str(uuid.uuid4()),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_version": "v1",
    "hyperparameters": json.dumps(grid.best_params_, sort_keys=True),
    "random_state": SEED,
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

existe = LOG_PATH.exists()

with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=experimento.keys()
    )

    if not existe or LOG_PATH.stat().st_size == 0:
        writer.writeheader()

    writer.writerow(experimento)