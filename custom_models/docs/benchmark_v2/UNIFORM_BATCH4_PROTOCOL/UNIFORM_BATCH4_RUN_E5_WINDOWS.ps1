$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$Python = "D:\Apps\Miniconda3\envs\env_tslib\python.exe"
$Profile = "uniform_train_batch4_v1"
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "custom_models\src"

Write-Warning "Deprecated: delegating to E5_CORE_28. Finalization is separate."
& (Join-Path $PSScriptRoot "UNIFORM_BATCH4_RUN_E5_CORE_28_WINDOWS.ps1")
exit $LASTEXITCODE
