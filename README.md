# mlflow-ev — ventas de vehículos eléctricos con MLflow

Mismo flujo que el ejemplo del profe (`ejemplo-uso-mlflow`), con otro conjunto de datos:
explorar, entrenar, registrar el mejor run con el alias `challenger` y predecir con ese alias.

**Problema:** predecir cuántos vehículos eléctricos (`value`) se venden en un país, un año,
un tipo de vehículo (`mode`) y un tipo de motor (`powertrain`). Es **regresión**, no clasificación,
así que la métrica es **R²** (también se guardan `r2_log`, `mae` y `rmse`), no accuracy.

## Qué cambia frente al ejemplo de crédito

| | Crédito (profe) | Vehículos eléctricos |
| --- | --- | --- |
| Objetivo | `loan_status` (0/1) | `value` (vehículos vendidos) |
| Filtros | edad 20–70, edad ≥ antigüedad | ventas históricas en vehículos, solo países, sin `EV` (= BEV + PHEV) |
| Texto → números | `LabelEncoder` | `OneHotEncoder` (61 países: con LabelEncoder, Colombia quedaría "al lado" de China) |
| Modelos | logística, SVM, árbol, random forest | lineal, SVR, árbol, random forest **regresores**, entrenados sobre log(1 + ventas) |
| Métrica para escoger | `accuracy` | `r2` |

`train.py`, `register.py`, `predict.py` y `MLproject` mantienen la estructura del profe.

## Estructura

```
├── README.md
├── requirements.txt          # mismas versiones que el ejemplo del profe
├── .env.example              # experimento ventas-ev, modelo modelo-ventas-ev
├── MLproject                 # main, register, predict, pipeline
├── iniciar.bat / iniciar.ps1 # todo el flujo en Windows con doble clic
├── notebooks/
│   └── 01_exploracion.ipynb  # filtros, codificación y comparación de los 4 modelos
├── src/
│   ├── config.py             # objetivo, filtros, métrica, semilla
│   ├── data.py               # cargar, filtrar, OneHotEncoder + StandardScaler, partir
│   ├── train.py              # 4 regresores, un run por modelo
│   ├── register.py           # mejor R² → alias challenger
│   └── predict.py            # carga por alias y predice ventas
└── data/
    └── EV_data_by_country_2026.csv
```

## Ejecutar

Windows: doble clic en `iniciar.bat`. Si existe `..\ejemplo-uso-mlflow\.venv`, lo reutiliza.

A mano (PowerShell, desde esta carpeta):

```powershell
..\ejemplo-uso-mlflow\.venv\Scripts\Activate.ps1   # o tu propio .venv
copy .env.example .env
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
$env:MLFLOW_EXPERIMENT_NAME="ventas-ev"
mlflow run . --env-manager local                 # entrenar
mlflow run . -e register --env-manager local     # mejor run -> challenger
mlflow run . -e predict --env-manager local      # carros BEV en Colombia
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Predecir con tu propio CSV (columnas `region_country, mode, powertrain, year`):

```powershell
python src/predict.py --input mis_casos.csv
```

## Resultados de referencia (test 30 %, semilla 42)

| Modelo | R² | R² log | MAE (vehículos) |
| --- | --- | --- | --- |
| Árbol de decisión | 0.957 | 0.804 | 17 478 |
| Random forest | 0.796 | 0.872 | 31 375 |
| Regresión lineal | 0.322 | 0.700 | 57 807 |
| SVR | 0.149 | 0.818 | 63 173 |

`r2` pesa mucho los países grandes (China); `r2_log` mide el orden de magnitud en todos.
Con `r2` gana el árbol; con `r2_log`, el random forest. Para cambiar el criterio basta
con `METRIC = "r2_log"` en `src/config.py`.
