param(
    [string]$PythonExe = "python"
)

# Fail on first error
$ErrorActionPreference = "Stop"

Write-Host "=== PaperLens OMR - Environment Setup ===" -ForegroundColor Cyan

# 1) Create venv if not exists
if (!(Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment (venv)..."
    & $PythonExe -m venv venv
} else {
    Write-Host "Virtual environment already exists."
}

# 2) Activate venv
$venvActivate = Join-Path $PWD "venv/Scripts/Activate.ps1"
Write-Host "Activating virtual environment..."
. $venvActivate

# 3) Upgrade pip/setuptools/wheel
Write-Host "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# 4) Install project requirements (compatible with Python 3.12 on Windows)
Write-Host "Installing project requirements from requirements.txt..."
pip install -r requirements.txt

Write-Host "\n=== Installed Packages (Top-Level) ===" -ForegroundColor Green
pip list --format=columns | Select-String -Pattern "streamlit|uvicorn|fastapi|sqlalchemy|numpy|scipy|opencv|pymupdf|pillow|pandas"

Write-Host "\n=== Next Steps ===" -ForegroundColor Yellow
Write-Host "To run the Streamlit app:" -ForegroundColor Yellow
Write-Host "  (venv) > python -m streamlit run app.py"
Write-Host "\nTo run the FastAPI server:" -ForegroundColor Yellow
Write-Host "  (venv) > python -m uvicorn fastapi_app:app --reload"

Write-Host "\nTips:" -ForegroundColor Yellow
Write-Host "- If PowerShell blocks this script, run:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
Write-Host "- If pip fails due to build tools, ensure you are on Python 3.12+ and retry. Wheels are provided for numpy/scipy/opencv on Windows."
