"""Carga el modelo por alias y predice.

Resuelve `models:/<MLFLOW_MODEL_NAME>@<MLFLOW_MODEL_ALIAS>`, aplica el
preprocesador guardado en ese run y devuelve las ventas estimadas
(número de vehículos eléctricos).

Es el entry point `predict` de `MLproject` (`mlflow run . -e predict`).
Sin `--input` usa carros BEV de Colombia del conjunto ya limpio.
"""

from __future__ import annotations

import argparse

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

import config
from data import aplicar_preprocesador, fila_cruda_ejemplo


def cargar():
    """Devuelve el estimador del alias y su preprocesador."""
    config.activar_seguimiento()
    cliente = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    version = cliente.get_model_version_by_alias(
        config.MLFLOW_MODEL_NAME,
        config.MLFLOW_MODEL_ALIAS,
    )
    modelo = mlflow.sklearn.load_model(version.source)
    ruta = mlflow.artifacts.download_artifacts(
        run_id=version.run_id,
        artifact_path="preprocessor.joblib",
        tracking_uri=config.MLFLOW_TRACKING_URI,
    )
    return modelo, joblib.load(ruta)


def predecir(datos: pd.DataFrame):
    """Predice una observación o un lote con las columnas originales."""
    modelo, preprocesador = cargar()
    features = aplicar_preprocesador(datos, preprocesador)
    return modelo.predict(features)


def main():
    parser = argparse.ArgumentParser(
        description="Predice ventas de vehículos eléctricos con el alias challenger."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="CSV con region_country, mode, powertrain y year. Si se omite, usa Colombia.",
    )
    args = parser.parse_args()
    datos = pd.read_csv(args.input) if args.input else fila_cruda_ejemplo()
    resultado = datos.copy()
    resultado["ventas_estimadas"] = predecir(datos).round().astype(int)
    print(resultado.to_string(index=False))


if __name__ == "__main__":
    main()
