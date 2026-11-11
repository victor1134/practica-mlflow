"""Registra el mejor run con el alias `challenger`.

Busca los runs de `MLFLOW_EXPERIMENT_NAME`, se queda con el de mayor
R² (la métrica del prototipo, `METRIC` en config.py) y publica ese modelo con el alias
`MLFLOW_MODEL_ALIAS`.

Es el entry point `register` de `MLproject` (`mlflow run . -e register`).
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

import config


def _uri_modelo(cliente: MlflowClient, run_id: str) -> str:
    """En MLflow 3 el estimador queda como logged model, no como carpeta del run."""
    run = cliente.get_run(run_id)
    salidas = []
    if run.outputs is not None and run.outputs.model_outputs:
        salidas = list(run.outputs.model_outputs)
    if not salidas:
        raise SystemExit(f"El run {run_id} no tiene un modelo registrado en MLflow.")
    elegido = max(salidas, key=lambda salida: salida.step)
    return f"models:/{elegido.model_id}"


def registrar():
    """Mueve el alias al run con mejor R²."""
    config.activar_seguimiento()
    cliente = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    experimento = cliente.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experimento is None:
        raise SystemExit(
            f"No existe el experimento {config.MLFLOW_EXPERIMENT_NAME}. "
            "Ejecuta antes mlflow run ."
        )

    runs = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string=(
            f"attributes.status = 'FINISHED' and metrics.{config.METRIC} > -1000000"
        ),
        order_by=[f"metrics.{config.METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(
            f"No hay runs con la métrica {config.METRIC}. Ejecuta antes mlflow run ."
        )

    mejor = runs[0]
    valor = mejor.data.metrics[config.METRIC]
    nombre = mejor.data.tags.get("mlflow.runName", mejor.info.run_id)
    version = mlflow.register_model(
        _uri_modelo(cliente, mejor.info.run_id),
        config.MLFLOW_MODEL_NAME,
    )
    cliente.set_registered_model_alias(
        config.MLFLOW_MODEL_NAME,
        config.MLFLOW_MODEL_ALIAS,
        version.version,
    )
    print(
        f"Mejor run: {nombre} ({config.METRIC}={valor:.6f}). "
        f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS} "
        f"-> versión {version.version}"
    )


if __name__ == "__main__":
    registrar()
