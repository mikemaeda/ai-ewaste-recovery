$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Opening plain Arducam viewer with automatic camera detection..."
Write-Host "Press Q or Esc in the camera window to quit."
python .\arducam_viewer.py --source auto --backend auto --width 1280 --height 720
