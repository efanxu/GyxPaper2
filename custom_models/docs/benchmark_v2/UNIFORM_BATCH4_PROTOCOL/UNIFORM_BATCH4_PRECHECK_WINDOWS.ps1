$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$Python = "D:\Apps\Miniconda3\envs\env_tslib\python.exe"
$Profile = "uniform_train_batch4_v1"
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "custom_models\src"

$active = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -match "^(python|pythonw|powershell|pwsh|cmd)(\.exe)?$") -and
  ($_.CommandLine -match "(run_benchmark\.py\s+(train|evaluate-only)|hardware[-_]preflight|run_st_mgprompt\.py.*--full|UNIFORM_BATCH4_RUN_)")
}
if ($active) { $active | Format-List; throw "BLOCKED_ACTIVE_FORMAL_RUN" }
$activeStatus = @()
Get-ChildItem -LiteralPath "custom_models" -Filter "run_status.json" -File -Recurse | ForEach-Object {
  try {
    $value = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($value.status -in @("RUNNING","STARTED","TRAINING","EVALUATING","PREFLIGHT_RUNNING") -and
        $value.formal_training -ne $false -and $value.run_mode -ne "smoke") {
      $activeStatus += $_.FullName
    }
  } catch {}
}
if ($activeStatus.Count -gt 0) { $activeStatus; throw "BLOCKED_ACTIVE_FORMAL_RUN" }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" protocol-check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "scripts/uniform_batch4_machine_gate.py" precheck
exit $LASTEXITCODE
