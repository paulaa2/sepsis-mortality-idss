$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$pipeline = Join-Path $repoRoot "IDSS\new_patient_pipeline.py"

Get-ChildItem -Path $scriptDir -Filter "patient_*.csv" | Sort-Object Name | ForEach-Object {
    Write-Host ""
    Write-Host "=== Ejecutando $($_.Name) ==="
    python $pipeline --patient-input $_.FullName
}
