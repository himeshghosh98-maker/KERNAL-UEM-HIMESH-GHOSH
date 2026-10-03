# GapDetect — one-command setup + run (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install -q -r requirements.txt

Write-Host "== Running tests =="
& ".\.venv\Scripts\python.exe" -m pytest -q
Write-Host "== Evaluation =="
& ".\.venv\Scripts\python.exe" evaluate.py
Write-Host "== Launching Streamlit on http://localhost:8501 =="
& ".\.venv\Scripts\streamlit.exe" run app.py --server.headless true
