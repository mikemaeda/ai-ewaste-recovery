$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Opening e-waste recovery dashboard with automatic camera detection..."
Write-Host "Press Q or Esc in the dashboard window to quit."
python .\detect_ewaste.py --source auto --backend auto --width 1280 --height 720 --conf 0.25
