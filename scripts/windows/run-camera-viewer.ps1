$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

Write-Host "Opening plain Arducam viewer with automatic camera detection..."
Write-Host "Press Q or Esc in the camera window to quit."
python -m ewaste_research.cli.view_camera --source auto --backend auto --width 1280 --height 720
