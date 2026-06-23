param(
    [string]$Geometry = 'tests/data/geometry.yaml',
    [string]$Output = 'plot.png'
)

# Run from repo root.
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\
Write-Host "Running visualizer with geometry: $Geometry -> $Output"

# Ensure the package src is on PYTHONPATH for imports.
$env:PYTHONPATH = 'src'

# Run the visualizer. Replace 'uv' with your preferred runner if needed.
$cmd = "uv run kinematics visualize --geometry $Geometry --output $Output"
Write-Host "Executing: $cmd"
try {
    iex $cmd
} catch {
    Write-Error "Failed to run visualizer: $_"
    exit 1
}
