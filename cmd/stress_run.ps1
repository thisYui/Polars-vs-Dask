$ErrorActionPreference = "Stop"

$ROOT = Resolve-Path "$PSScriptRoot\.."
Set-Location $ROOT

$commands = @(
    "python .\scripts\run_pipeline.py --steps benchmark_pandas --sizes 10M_skewed 10M_highuid --data-type syn --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_polars --sizes 10M_skewed 10M_highuid --data-type syn --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_polars_eager --sizes 10M_skewed 10M_highuid --data-type syn --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_dask --sizes 10M_skewed 10M_highuid --data-type syn --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_dask --sizes 100M --data-type syn --workloads filter groupby --verbose",
    "python .\scripts\run_pipeline.py --steps benchmark_polars --sizes 100M --data-type syn --workloads filter groupby --verbose"   
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