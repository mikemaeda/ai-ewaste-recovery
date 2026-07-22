# scripts/windows/setup-yolo.ps1
# One-shot installer for the YOLO11 e-waste detection system on Windows + PowerShell.
#
# HOW TO RUN (from VS Code's PowerShell terminal, inside the project folder):
#
#     powershell -ExecutionPolicy Bypass -File .\scripts/windows/setup-yolo.ps1
#
# It installs every dependency into whatever Python is currently active. If you
# use a virtual environment, activate it FIRST, then run this script.

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " e-waste AI - YOLO detection setup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Confirm Python is available and show its version.
Write-Host "`n[1/3] Checking for Python..." -ForegroundColor Yellow
try {
    $pythonVersion = (python --version) 2>&1
    Write-Host "      Found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python was not found on your PATH." -ForegroundColor Red
    Write-Host "Install Python from https://www.python.org/downloads/ and tick" -ForegroundColor Red
    Write-Host "'Add Python to PATH' during installation, then re-run this script." -ForegroundColor Red
    exit 1
}

# 2. Upgrade pip so the heavy ML wheels (torch) install cleanly.
Write-Host "`n[2/3] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 3. Install everything from requirements_yolo.txt.
Write-Host "`n[3/3] Installing YOLO dependencies (this downloads PyTorch, ~minutes)..." -ForegroundColor Yellow
python -m pip install -r requirements_yolo.txt

Write-Host "`n==============================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host "`nTry the live detection window right now (no trained model needed):" -ForegroundColor Cyan
Write-Host "    python -m ewaste_research.cli.run_detector --source 0" -ForegroundColor White
Write-Host "`nOr run it on a single image:" -ForegroundColor Cyan
Write-Host "    python -m ewaste_research.cli.run_detector --source path\to\photo.jpg" -ForegroundColor White
Write-Host "`nPress Q in the window to quit." -ForegroundColor Cyan
