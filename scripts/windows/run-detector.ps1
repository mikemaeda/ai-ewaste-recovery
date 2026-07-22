$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

Write-Host "Opening e-waste recovery dashboard with automatic camera detection..."
Write-Host "Press Q or Esc in the dashboard window to quit."
python -m ewaste_research.cli.run_detector --source auto --backend auto --width 1280 --height 720 --conf 0.25
