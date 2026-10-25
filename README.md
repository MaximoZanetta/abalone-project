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

`git clone https://github.com/MaximoZanetta/abalone-project.git`

Ingresar al directorio del proyecto:

`cd abalone-project`

### 2. Instalar uv

Es necesario tener instalado uv para administrar las dependencias del proyecto.

### 3. Crear y sincronizar el entorno

Ejecutar:

`uv sync --locked`

Este comando recrea el entorno virtual utilizando las versiones de dependencias registradas en `uv.lock`.

### 4. Verificar la versión de Python

`uv run python --version`

### 5. Ejecutar el proyecto

Ver las secciones **Preprocesamiento** y **Entrenamiento y evaluación del modelo** más abajo.

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

## Preprocesamiento

Para generar `data/processed/abalone_clean.csv` y `data/data_log.json` a partir de `data/raw/abalone_raw.csv`:

```bash
uv run python src/preprocess.py
```

El script:

- Elimina registros físicamente inconsistentes (`height <= 0`, `height > length`, `shellweight > wholeweight`). Ese es el criterio declarado de valores atípicos.
- Crea `age = rings + 1.5`.
- Codifica `sex` con one-hot (`sex_F`, `sex_I`, `sex_M`).
- No escala variables: el escalado, de ser necesario, va dentro del Pipeline de entrenamiento para no filtrar información del test.
- Registra en `data/data_log.json` la fecha, el hash SHA-256 del archivo original, los parámetros, las versiones de Python/pandas y las filas eliminadas.

Cada vez que se modifica `preprocess.py`, el CSV procesado se regenera y se commitea en el mismo commit.

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

## Reproducibilidad del registro de datos

**¿Qué información de `data_log.json` es imprescindible para reconstruir el dataset?** El archivo de origen y una forma de verificar que es el mismo (hash SHA-256), cada transformación con sus parámetros exactos y en orden (fórmula de `age`, reglas de outliers con sus umbrales, tipo de codificación) y las versiones de Python y pandas. Opcionalmente, los índices eliminados sirven para comprobar el resultado.

**¿Alcanza el registro actual?** Con el hash, los parámetros y las versiones registradas, sí alcanza para reconstruirlo, siempre que el código de `preprocess.py` del mismo commit esté disponible. Aun así, el registro describe los pasos en texto y no los ejecuta: la reconstrucción depende de que el script se mantenga sincronizado con el log. Antes de este ajuste faltaban el hash y los parámetros, y el timestamp hacía que el log cambiara en cada ejecución.

**¿Y si el dataset pesara varios GB o cambiara cada semana?** Git guarda cada versión de los archivos binarios o grandes completa en el historial, así que el repositorio crecería sin control, los clones serían lentos y podrían superarse los límites de GitHub. Con cambios semanales cada versión se duplicaría en el historial y los diffs no serían legibles. La alternativa es versionar solo metadatos y hashes en Git y guardar los datos en un almacenamiento externo (por ejemplo DVC, Git LFS o un bucket), de modo que cada commit apunte a una versión concreta del dato.

## Resultados

- **v2** (`exp/hiperparametros`, Random Forest con grilla ampliada: `max_depth=8`, `min_samples_leaf=8`, `max_features=1.0`, `n_estimators=200`): MAE 1.4839, RMSE 2.1485, R² 0.5972 (mejora levemente sobre v1: MAE 1.5032).
