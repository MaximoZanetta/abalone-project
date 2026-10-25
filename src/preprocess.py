
import hashlib
import json
import platform
from pathlib import Path
from datetime import date

import pandas as pd

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data/raw/abalone_raw.csv"
CLEAN_PATH = BASE_DIR / "data/processed/abalone_clean.csv"
LOG_PATH = BASE_DIR / "data/data_log.json"

# Criterio de valores atípicos: se eliminan registros físicamente imposibles
# (altura nula/negativa, altura mayor al largo, peso de concha mayor al peso total).
# No se usan umbrales estadísticos: solo reglas de consistencia física.
OUTLIER_RULES = [
    "height <= 0",
    "height > length",
    "shellweight > wholeweight",
]

# 1. Cargar dataset original
columnas = [
    "sex",
    "length",
    "diameter",
    "height",
    "wholeweight",
    "shuckedweight",
    "visceraweight",
    "shellweight",
    "rings"
]

# 2. Cargar dataset sin encabezado
df = pd.read_csv(
    RAW_PATH,
    header=None,
    names=columnas
)

print("Filas originales:", len(df))
print(df.head())

# Normalizar nombres de columnas
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "", regex=False)
    .str.replace("_", "", regex=False)
)

# Verificar columnas necesarias
columnas = {
    "sex", "rings", "height",
    "length", "shellweight", "wholeweight"
}

faltantes = columnas - set(df.columns)

if faltantes:
    raise ValueError(f"Faltan columnas: {faltantes}")

# 2. Detectar registros físicamente inconsistentes
inconsistentes = (
    (df["height"] <= 0)
    | (df["height"] > df["length"])
    | (df["shellweight"] > df["wholeweight"])
)

indices_eliminados = df.index[inconsistentes].tolist()
cantidad_eliminada = int(inconsistentes.sum())

print("Registros eliminados:", cantidad_eliminada)

df = df.loc[~inconsistentes].copy()

# 3. Crear la edad estimada
df["age"] = df["rings"] + 1.5

# 4. One-Hot Encoding de sex
df = pd.get_dummies(
    df,
    columns=["sex"],
    prefix="sex",
    dtype=int
)

# 5. Guardar dataset procesado
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)

# 6. Registrar las transformaciones
registro = {
    "version": "v1",
    "created_at": date.today().isoformat(),
    "source": "data/raw/abalone_raw.csv",
    "source_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
    "environment": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
    },
    "parameters": {
        "age_formula": "age = rings + 1.5",
        "sex_encoding": "one-hot (sex_F, sex_I, sex_M), dtype int",
        "outlier_rules": OUTLIER_RULES,
        "scaling": "none (se aplica en train_model.py dentro del Pipeline)",
    },
    "output": "data/processed/abalone_clean.csv",
    "original_rows": len(df) + cantidad_eliminada,
    "processed_rows": len(df),
    "removed_rows": cantidad_eliminada,
    "removed_indices": indices_eliminados,
    "transformations": [
        "Removed rows with height <= 0",
        "Removed rows with height > length",
        "Removed rows with shellweight > wholeweight",
        "Created age = rings + 1.5",
        "One-hot encoded sex into sex_F, sex_I, sex_M"
    ]
}

with open(LOG_PATH, "w", encoding="utf-8") as archivo:
    json.dump(registro, archivo, indent=4, ensure_ascii=False)

print("Preprocesamiento finalizado.")
print("Filas procesadas:", len(df))
print("Archivo generado:", CLEAN_PATH)
