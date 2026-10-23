# Abalone Project — Laboratorio de Datos II

## Descripción del proyecto

Este proyecto corresponde al Trabajo Práctico 5 de Laboratorio de Datos II, de la carrera Ingeniería en Inteligencia Artificial.

El objetivo es desarrollar un proyecto de Machine Learning reproducible utilizando el dataset **Abalone**, que contiene mediciones físicas de moluscos marinos.

Se entrenará un modelo de regresión para predecir la edad estimada del abalón a partir de sus características físicas.

El trabajo se enfoca en la reproducibilidad, el versionado de código y datos, el registro de experimentos y el trabajo colaborativo mediante Git.

## Tecnologías utilizadas

- **Python:** 3.13
- **uv:** gestión del entorno virtual y dependencias.
- **Git:** control de versiones.
- **Pandas y NumPy:** procesamiento de datos.
- **Matplotlib y Seaborn:** visualización.
- **Scikit-learn:** entrenamiento y evaluación de modelos.
- **Joblib:** serialización de modelos.

## Instalación y configuración

### 1. Clonar el repositorio

`git clone URL_DEL_REPOSITORIO`

Ingresar al directorio del proyecto:

`cd abalone_project`

### 2. Instalar uv

Es necesario tener instalado uv para administrar las dependencias del proyecto.

### 3. Crear y sincronizar el entorno

Ejecutar:

`uv sync --locked`

Este comando recrea el entorno virtual utilizando las versiones de dependencias registradas en `uv.lock`.

### 4. Verificar la versión de Python

`uv run python --version`

### 5. Ejecutar el proyecto

Los comandos para ejecutar el preprocesamiento y entrenar el modelo se agregarán en los ejercicios siguientes.

## Estructura del proyecto

- **data/raw/**: dataset original sin modificaciones.
- **data/processed/**: dataset limpio y transformado.
- **data/data_log.json**: registro de las transformaciones realizadas.
- **src/preprocess.py**: script de preprocesamiento.
- **src/train_model.py**: script de entrenamiento y evaluación.
- **results/**: métricas y registro de experimentos.
- **models/**: modelos entrenados y serializados.
- **pyproject.toml**: dependencias y configuración del proyecto.
- **uv.lock**: versiones resueltas de las dependencias.
- **requirements.txt**: listado de dependencias.
- **.gitignore**: archivos excluidos del control de versiones.

## Reproducibilidad

El proyecto utiliza uv para mantener un entorno consistente y Git para registrar los cambios.

Las transformaciones de datos, las semillas aleatorias, los hiperparámetros y las métricas de los modelos serán documentados para facilitar la reproducción de los experimentos.

## Entrenamiento y evaluación del modelo

Para entrenar el modelo de regresión sobre el dataset Abalone procesado, ejecutar:

```bash
uv run python src/train_model.py
```

El script:

- Carga el dataset procesado.
- Utiliza `age` como variable objetivo y excluye `rings` para evitar data leakage.
- Divide los datos en entrenamiento (80%) y test (20%), con semilla fija.
- Entrena un `RandomForestRegressor` mediante un Pipeline de scikit-learn.
- Evalúa el modelo utilizando MAE, RMSE y R².
- Guarda el modelo comprimido en `models/model_v1.pkl`.
- Registra las métricas e hiperparámetros en `results/metrics_v1.json`.
- Agrega una entrada al historial de experimentos en `results/experiments_log.csv`.

El archivo de experimentos registra cada ejecución con un identificador y una marca temporal, por lo que cambia cada vez que se ejecuta el entrenamiento.