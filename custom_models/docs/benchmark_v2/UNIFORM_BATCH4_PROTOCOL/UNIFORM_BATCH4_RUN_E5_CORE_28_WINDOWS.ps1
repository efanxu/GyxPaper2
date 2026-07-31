$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$Python = "D:\Apps\Miniconda3\envs\env_tslib\python.exe"
$Profile = "uniform_train_batch4_v1"
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "custom_models\src"

& $Python "scripts/uniform_batch4_machine_gate.py" formal-exec --suite e5_core --python "$Python"
exit $LASTEXITCODE
