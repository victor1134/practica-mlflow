"""Entrena las cuatro estrategias del notebook y las deja en MLflow.

Cada modelo es un run del experimento `MLFLOW_EXPERIMENT_NAME`:
regresión lineal, SVR, árbol de decisión y random forest, todos de
regresión. Las ventas van de 1 a cientos de miles de vehículos, así que
cada modelo aprende sobre log(1 + ventas) y devuelve vehículos
(`TransformedTargetRegressor`). En todos se registran R², MAE y RMSE
sobre la partición de prueba, los parámetros y el
estimador, junto con el preprocesador necesario para predecir. La
columna Dataset del run apunta a la partición de entrenamiento y a la
de prueba.

Es el entry point `main` de `MLproject` (`mlflow run .`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import mlflow.sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import config
from data import cargar_particion, registrar_preprocesamiento


def _en_log(regresor):
    """Entrena sobre log(1 + ventas) y devuelve la predicción en vehículos."""
    return TransformedTargetRegressor(regressor=regresor, func=np.log1p, inverse_func=np.expm1)


def _modelos():
    semilla = config.RANDOM_SEED
    return [
        ("regresion_lineal", _en_log(LinearRegression())),
        ("svr", _en_log(SVR())),
        ("arbol_decision", _en_log(DecisionTreeRegressor(random_state=semilla))),
        ("random_forest", _en_log(RandomForestRegressor(random_state=semilla))),
    ]


def _metricas(y_real, y_pred) -> dict:
    """R² (la que decide), errores en vehículos y R² en escala logarítmica."""
    return {
        "r2": r2_score(y_real, y_pred),
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "r2_log": r2_score(np.log1p(y_real), np.log1p(np.clip(y_pred, 0, None))),
    }


def _datasets(x_train, y_train, x_test, y_test):
    """Arma los dos conjuntos que muestra la columna Dataset de cada run."""
    origen = str(config.DATA_PATH)

    def conjunto(features, objetivo, nombre):
        tabla = features.copy()
        tabla[config.TARGET] = objetivo
        return mlflow.data.from_pandas(
            tabla,
            source=origen,
            targets=config.TARGET,
            name=nombre,
        )

    return (
        conjunto(x_train, y_train, "ventas_ev_train"),
        conjunto(x_test, y_test, "ventas_ev_test"),
    )


def _parametros(modelo) -> dict:
    parametros = {}
    for clave, valor in modelo.get_params().items():
        if isinstance(valor, (int, float, str, bool)):
            parametros[clave] = valor
        else:
            parametros[clave] = "None" if valor is None else str(valor)
    return parametros


def entrenar():
    """Ajusta los cuatro modelos y abre un run por cada uno."""
    # `mlflow run` ya abrió un run padre. Los modelos van en runs propios
    # del experimento configurado, para que `register.py` pueda compararlos.
    os.environ.pop("MLFLOW_RUN_ID", None)
    config.activar_seguimiento()
    mlflow.sklearn.autolog(disable=True)

    x_train, x_test, y_train, y_test, preprocesador, material = cargar_particion()
    with mlflow.start_run(run_name="preprocesamiento"):
        registrar_preprocesamiento(material)
    print(
        f"Partición: train {x_train.shape[0]} filas, "
        f"test {x_test.shape[0]} filas, {x_train.shape[1]} features"
    )

    datos_train, datos_test = _datasets(x_train, y_train, x_test, y_test)

    with tempfile.TemporaryDirectory() as tmp:
        ruta_pre = Path(tmp) / "preprocessor.joblib"
        joblib.dump(preprocesador, ruta_pre)

        for nombre, modelo in _modelos():
            with mlflow.start_run(run_name=nombre):
                mlflow.log_input(datos_train, context="training")
                mlflow.log_input(datos_test, context="evaluation")
                modelo.fit(x_train, y_train)
                prediccion = modelo.predict(x_test)
                metricas = _metricas(y_test, prediccion)

                mlflow.log_params(_parametros(modelo))
                mlflow.log_param("estrategia", nombre)
                mlflow.log_param("test_size", config.TEST_SIZE)
                mlflow.log_param("random_seed", config.RANDOM_SEED)
                mlflow.log_metrics(metricas)
                mlflow.log_artifact(str(ruta_pre))
                mlflow.sklearn.log_model(
                    modelo,
                    name="model",
                    input_example=x_train.head(2),
                    serialization_format="pickle",
                )
                print(
                    f"{nombre}: r2={metricas['r2']:.4f}  r2_log={metricas['r2_log']:.4f}  "
                    f"mae={metricas['mae']:,.0f}  rmse={metricas['rmse']:,.0f} vehículos"
                )


if __name__ == "__main__":
    entrenar()
