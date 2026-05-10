$ErrorActionPreference = "Stop"

$ROOT = Resolve-Path "$PSScriptRoot\.."
Set-Location $ROOT

$commands = @(
    "python .\scripts\run_pipeline.py --steps benchmark_pandas --sizes 5GB 10GB 20GB --data-type syn --partition --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_polars --sizes 5GB 10GB 20GB --data-type syn --partition --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_polars_eager --sizes 5GB 10GB 20GB --data-type syn --partition --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_dask --sizes 5GB 10GB 20GB --data-type syn --partition --verbose"
)

foreach ($cmd in $commands) {
    Write-Host "`n=== Running: $cmd ===" -ForegroundColor Cyan
    Invoke-Expression $cmd

    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nCommand failed. Stop." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Start-Sleep -Seconds 5
}

Write-Host "`nAll benchmark commands completed." -ForegroundColor Green