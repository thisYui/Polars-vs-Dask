$ErrorActionPreference = "Stop"

$ROOT = Resolve-Path "$PSScriptRoot\.."
Set-Location $ROOT

$commands = @(
    "python .\scripts\run_pipeline.py --steps compress --partition --verbose",
    "python .\scripts\run_pipeline.py --steps preprocess --partition --verbose",
    "python .\scripts\run_pipeline.py --steps split_real --sizes 1M 10M 50M --data-type real --partition --verbose"
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