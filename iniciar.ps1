# iniciar.ps1 - Ejecuta el proyecto de ventas de vehiculos electricos con MLflow (Windows PowerShell)
# Doble clic en iniciar.bat, o:  powershell -ExecutionPolicy Bypass -File .\iniciar.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Pausa($msg) { Write-Host $msg -ForegroundColor Red; Read-Host "Enter para salir"; exit 1 }

# 1) Entorno: el propio .venv, o el que ya se instalo en ..\ejemplo-uso-mlflow (mismas versiones)
if (Test-Path ".venv\Scripts\Activate.ps1") { $venv = ".venv" }
elseif (Test-Path "..\ejemplo-uso-mlflow\.venv\Scripts\Activate.ps1") { $venv = "..\ejemplo-uso-mlflow\.venv"; Write-Host "== Reutilizando el entorno de ejemplo-uso-mlflow ==" -ForegroundColor Cyan }
else {
    Write-Host "== Creando entorno virtual .venv ==" -ForegroundColor Cyan
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv .venv 2>$null
        if (-not (Test-Path ".venv\Scripts\python.exe")) { py -3 -m venv .venv }
    } elseif (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv }
    else { Pausa "No encuentro Python. Instala Python 3.12 desde python.org (marca 'Add to PATH')." }
    $venv = ".venv"
}
& "$venv\Scripts\Activate.ps1"
python --version
pip install -q -r requirements.txt

# 2) .env local
if (-not (Test-Path ".env")) { Copy-Item .env.example .env }
Get-Content .env | Where-Object { $_ -match '^\s*[A-Z_]+=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim()
}
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
mlflow --version
if (-not (Test-Path $env:DATA_PATH)) { Pausa "Falta el dataset: $env:DATA_PATH" }

# 3) Pipeline: entrenar 4 regresores, registrar challenger, predecir
Write-Host "== mlflow run . (entrenar) ==" -ForegroundColor Cyan
mlflow run . --env-manager local
Write-Host "== mlflow run . -e register ==" -ForegroundColor Cyan
mlflow run . -e register --env-manager local
Write-Host "== mlflow run . -e predict (carros BEV en Colombia) ==" -ForegroundColor Cyan
mlflow run . -e predict --env-manager local

# 4) Interfaz (si el 5000 esta ocupado por la otra ventana, usa el 5001)
$puerto = 5000
if (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue) { $puerto = 5001 }
Write-Host "== Interfaz en http://127.0.0.1:$puerto (Ctrl+C para cerrar) ==" -ForegroundColor Green
mlflow ui --backend-store-uri $env:MLFLOW_TRACKING_URI --port $puerto
