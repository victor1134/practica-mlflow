"""Cargar, limpiar y partir los datos de ventas de vehículos eléctricos.

Sigue `notebooks/01_exploracion.ipynb`:

- deja solo las ventas históricas de EV, en vehículos, por país;
- quita el powertrain "EV" (suma de BEV + PHEV) y las ventas menores que 1;
- elimina las columnas que quedan constantes, duplicados y filas incompletas;
- codifica las columnas de texto con `OneHotEncoder` (una columna por país,
  modo y powertrain; con LabelEncoder el modelo creería que el país 9
  "está cerca" del país 8);
- escala el resto de features con `StandardScaler`, sin tocar el objetivo;
- parte entrenamiento y prueba con `TEST_SIZE` y `RANDOM_SEED`.

El codificador y el escalador se ajustan con todo el conjunto limpio,
igual que en el cuaderno, y después se hace la partición.

`registrar_preprocesamiento` deja en el run activo los conteos de cada
filtro, las categorías de `OneHotEncoder`, la media y la escala de
`StandardScaler`, y una muestra antes y después de transformar.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

MUESTRA = 10


def _leer_csv() -> pd.DataFrame:
    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No está el conjunto en {config.DATA_PATH}. "
            "Revisa DATA_PATH en .env."
        )
    datos = pd.read_csv(config.DATA_PATH)
    necesarias = (config.TARGET, *config.CATEGORICAS, *config.COLUMNS_TO_DROP, "year")
    faltan = [c for c in necesarias if c not in datos.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {config.DATA_PATH.name}: {faltan}")
    return datos


def _limpiar(datos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica los filtros del notebook y cuenta cuántas filas quita cada uno."""
    resumen = {
        "archivo": config.DATA_PATH.name,
        "filas_leidas": int(len(datos)),
        "columnas_leidas": list(datos.columns),
    }

    n = len(datos)
    datos = datos[
        (datos["category"] == config.CATEGORIA)
        & (datos["parameter"] == config.PARAMETRO)
        & (datos["unit"] == config.UNIDAD)
    ]
    resumen["filas_otros_indicadores"] = int(n - len(datos))

    n = len(datos)
    datos = datos[datos["Aggregate group"].isin(config.GRUPOS_PAIS)]
    resumen["filas_regiones_agregadas"] = int(n - len(datos))

    n = len(datos)
    datos = datos[~datos["powertrain"].isin(config.POWERTRAIN_EXCLUIDOS)]
    resumen["filas_powertrain_ev_total"] = int(n - len(datos))

    datos = datos.drop(columns=list(config.COLUMNS_TO_DROP))
    resumen["columnas_eliminadas"] = list(config.COLUMNS_TO_DROP)

    n = len(datos)
    datos = datos.drop_duplicates()
    resumen["filas_duplicadas"] = int(n - len(datos))

    n = len(datos)
    datos = datos.dropna()
    resumen["filas_incompletas"] = int(n - len(datos))

    n = len(datos)
    datos = datos[datos[config.TARGET] >= config.VALOR_MIN]
    resumen["filas_valor_menor_que_minimo"] = int(n - len(datos))

    datos = datos.reset_index(drop=True)
    resumen["filas_limpias"] = int(len(datos))
    resumen["paises"] = int(datos["region_country"].nunique())
    resumen["anio_min"] = int(datos["year"].min())
    resumen["anio_max"] = int(datos["year"].max())
    return datos, resumen


def cargar_limpio() -> pd.DataFrame:
    """Lee `DATA_PATH` y aplica los filtros del notebook."""
    datos, _ = _limpiar(_leer_csv())
    return datos


def ajustar_preprocesador(datos: pd.DataFrame) -> dict:
    """Ajusta el codificador y el escalador. No incluye el objetivo."""
    entrada = [columna for columna in datos.columns if columna != config.TARGET]
    categoricas = list(config.CATEGORICAS)
    numericas = [columna for columna in entrada if columna not in categoricas]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(datos[categoricas].astype(str))
    scaler = StandardScaler()
    scaler.fit(datos[numericas])

    features = list(encoder.get_feature_names_out(categoricas)) + numericas
    return {
        "encoder": encoder,
        "scaler": scaler,
        "categoricas": categoricas,
        "numericas": numericas,
        "entrada": entrada,
        "features": features,
    }


def aplicar_preprocesador(datos: pd.DataFrame, preprocesador: dict) -> pd.DataFrame:
    """Devuelve las features codificadas y escaladas, en el orden de entrenamiento."""
    faltan = [c for c in preprocesador["entrada"] if c not in datos.columns]
    if faltan:
        raise ValueError(f"Faltan columnas para predecir: {faltan}")

    categoricas = preprocesador["categoricas"]
    numericas = preprocesador["numericas"]
    codificadas = pd.DataFrame(
        preprocesador["encoder"].transform(datos[categoricas].astype(str)),
        columns=preprocesador["encoder"].get_feature_names_out(categoricas),
        index=datos.index,
    )
    escaladas = pd.DataFrame(
        preprocesador["scaler"].transform(datos[numericas]),
        columns=numericas,
        index=datos.index,
    )
    return pd.concat([codificadas, escaladas], axis=1)[preprocesador["features"]]


def _transformaciones(preprocesador: dict) -> dict:
    """Mapa legible del codificador y del escalador, sin el objeto de sklearn."""
    codificacion = {
        columna: [str(valor) for valor in categorias]
        for columna, categorias in zip(
            preprocesador["categoricas"], preprocesador["encoder"].categories_
        )
    }
    scaler = preprocesador["scaler"]
    escalado = {
        columna: {"media": float(scaler.mean_[i]), "escala": float(scaler.scale_[i])}
        for i, columna in enumerate(preprocesador["numericas"])
    }
    return {
        "objetivo": config.TARGET,
        "test_size": config.TEST_SIZE,
        "random_seed": config.RANDOM_SEED,
        "codificacion_onehot": codificacion,
        "escalado_standard": escalado,
        "features": list(preprocesador["features"]),
        "pasos": [
            f"category={config.CATEGORIA}, parameter={config.PARAMETRO}, unit={config.UNIDAD}",
            f"Aggregate group en {', '.join(config.GRUPOS_PAIS)} (solo países)",
            f"sin powertrain {', '.join(config.POWERTRAIN_EXCLUIDOS)}",
            f"eliminar columnas: {', '.join(config.COLUMNS_TO_DROP)}",
            "quitar duplicados y filas incompletas",
            f"{config.TARGET} >= {config.VALOR_MIN}",
            "OneHotEncoder en region_country, mode y powertrain",
            "StandardScaler en year",
            f"partición test_size={config.TEST_SIZE}, random_state={config.RANDOM_SEED}",
        ],
    }


def cargar_particion():
    """Devuelve train, test, el preprocesador y el material para MLflow."""
    crudo = _leer_csv()
    datos, resumen = _limpiar(crudo)
    preprocesador = ajustar_preprocesador(datos)
    features = aplicar_preprocesador(datos, preprocesador)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        datos[config.TARGET],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    material = {
        "resumen": resumen,
        "transformaciones": _transformaciones(preprocesador),
        "muestra_cruda": crudo.head(MUESTRA),
        "muestra_limpia": datos.head(MUESTRA),
        "muestra_transformada": features.head(MUESTRA),
        "filas_train": int(len(x_train)),
        "filas_test": int(len(x_test)),
    }
    return x_train, x_test, y_train, y_test, preprocesador, material


def registrar_preprocesamiento(material: dict) -> None:
    """Escribe filtros, transformaciones y muestras en el run activo."""
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError("registrar_preprocesamiento necesita un run de MLflow activo.")

    resumen = material["resumen"]
    mlflow.log_params(
        {
            "archivo": resumen["archivo"],
            "filas_leidas": resumen["filas_leidas"],
            "filas_duplicadas": resumen["filas_duplicadas"],
            "filas_incompletas": resumen["filas_incompletas"],
            "filas_otros_indicadores": resumen["filas_otros_indicadores"],
            "filas_regiones_agregadas": resumen["filas_regiones_agregadas"],
            "filas_powertrain_ev_total": resumen["filas_powertrain_ev_total"],
            "filas_valor_menor_que_minimo": resumen["filas_valor_menor_que_minimo"],
            "filas_limpias": resumen["filas_limpias"],
            "filas_train": material["filas_train"],
            "filas_test": material["filas_test"],
            "paises": resumen["paises"],
            "anio_min": resumen["anio_min"],
            "anio_max": resumen["anio_max"],
            "columnas_eliminadas": ",".join(resumen["columnas_eliminadas"]),
            "columnas_categoricas": ",".join(config.CATEGORICAS),
        }
    )
    mlflow.log_dict(
        {"limpieza": resumen, **material["transformaciones"]},
        "preprocesamiento/transformaciones.json",
    )
    mlflow.log_table(material["muestra_cruda"], "preprocesamiento/muestra_cruda.json")
    mlflow.log_table(material["muestra_limpia"], "preprocesamiento/muestra_limpia.json")
    mlflow.log_table(
        material["muestra_transformada"],
        "preprocesamiento/muestra_transformada.json",
    )


def fila_cruda_ejemplo() -> pd.DataFrame:
    """Observaciones sin el objetivo, para probar `predict` sin un archivo extra:
    carros eléctricos (BEV) en Colombia, los últimos años disponibles."""
    datos = cargar_limpio()
    ejemplo = datos[
        (datos["region_country"] == "Colombia")
        & (datos["mode"] == "Cars")
        & (datos["powertrain"] == "BEV")
    ].tail(3)
    if ejemplo.empty:
        ejemplo = datos.tail(1)
    return ejemplo.drop(columns=[config.TARGET])
