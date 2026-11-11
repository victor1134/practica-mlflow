"""Nombres del experimento y del modelo, semilla y decisiones del prototipo.

`src/` lee estos valores desde aquí. Salen de `.env`; si falta el
archivo, se usan los valores por defecto de abajo.

El objetivo, los filtros y la métrica salen del prototipo de
`notebooks/01_exploracion.ipynb`: predecir `value` (número de vehículos
eléctricos vendidos en un país, un año, un tipo de vehículo y un tipo de
motor) y elegir el run con mayor R².
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ventas-ev")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "modelo-ventas-ev")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

_ruta_datos = os.getenv("DATA_PATH", "data/EV_data_by_country_2026.csv")
DATA_PATH = (
    Path(_ruta_datos)
    if Path(_ruta_datos).is_absolute()
    else RAIZ / _ruta_datos
)

# Decisiones cerradas en el notebook. No se repiten en el resto de src/.
TARGET = "value"                 # vehículos vendidos
TEST_SIZE = 0.3
METRIC = "r2"                    # métrica con la que register.py escoge el mejor run

# Filtros: de las ~49 000 filas solo sirven las ventas históricas de EV, por país.
CATEGORIA = "Historical"         # fuera las proyecciones (STEPS, CPS)
PARAMETRO = "EV sales"           # fuera stock, precios, cargadores, participación...
UNIDAD = "Vehicles"
GRUPOS_PAIS = ("Other", "Projection_country")      # países; fuera World, EU, agregados
POWERTRAIN_EXCLUIDOS = ("EV",)   # "EV" es la suma de BEV + PHEV: repetiría información
VALOR_MIN = 1                    # al menos un vehículo vendido

CATEGORICAS = (
    "region_country",
    "mode",
    "powertrain",
)

# Columnas que quedan constantes después de filtrar (no aportan nada al modelo).
COLUMNS_TO_DROP = (
    "category",
    "parameter",
    "unit",
    "Aggregate group",
)


def activar_seguimiento():
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
