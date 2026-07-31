param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$Python = "D:\Apps\Miniconda3\envs\env_tslib\python.exe"
$Profile = "uniform_train_batch4_v1"
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "custom_models\src"
if (-not $env:UNIFORM_BATCH4_GIT_COMMIT) {
    $env:UNIFORM_BATCH4_GIT_COMMIT = "8c1ee2881b88bfa81b0bbfed588fc7976e4e2063"
}

$FormalArgs = @(
    "scripts/uniform_batch4_machine_gate.py",
    "formal-exec",
    "--suite", "original_skip_msgnet",
    "--python", "$Python"
)
if ($DryRun) {
    $FormalArgs += "--dry-run"
}
& $Python @FormalArgs
exit $LASTEXITCODE
