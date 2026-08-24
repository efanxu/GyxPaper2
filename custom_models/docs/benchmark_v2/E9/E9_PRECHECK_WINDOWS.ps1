[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\PaperProject\GyxPaper2',
    [string]$Python = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$AnalysisRoot,
    [string]$Original26Xlsx
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) { throw "Missing ProjectRoot: $ProjectRoot" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
if ([string]::IsNullOrWhiteSpace($AnalysisRoot)) {
    $AnalysisRoot = Join-Path $ProjectRoot 'custom_models\results\benchmark_v2\msmg_dwu_transfer_seed2026'
}
$Manifest = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\E9\E9_VARIANT_MANIFEST.json'
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw "Missing E9 variant manifest: $Manifest" }
if (-not [string]::IsNullOrWhiteSpace($Original26Xlsx) -and -not (Test-Path -LiteralPath $Original26Xlsx -PathType Leaf)) {
    throw "Missing original26 workbook: $Original26Xlsx"
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'custom_models\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath $ProjectRoot
. (Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\WINDOWS_NATIVE_PROCESS_RUNNER.ps1')

function Invoke-VisiblePython {
    param([string[]]$Arguments)
    $Native = Invoke-GyxNativeProcess -FilePath $Python -Arguments $Arguments -WorkingDirectory $ProjectRoot
    if (-not [string]::IsNullOrWhiteSpace([string]$Native.Stdout)) { Write-Host ([string]$Native.Stdout).TrimEnd() }
    if (-not [string]::IsNullOrWhiteSpace([string]$Native.Stderr)) { Write-Host -ForegroundColor DarkYellow ([string]$Native.Stderr).TrimEnd() }
    if ($Native.ExitCode -ne 0) { exit $Native.ExitCode }
}

Write-Host "ProjectRoot=$ProjectRoot"
Write-Host "AnalysisRoot=$AnalysisRoot"
Write-Host ('git HEAD=' + (& git rev-parse HEAD))
Write-Host ('git branch=' + (& git branch --show-current))

$EnvironmentProbe = @'
import sys, torch
print(f"Python executable={sys.executable}")
print(f"Python version={sys.version.split()[0]}")
print(f"torch version={torch.__version__}")
print(f"CUDA available={torch.cuda.is_available()}")
print(f"CUDA version={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit(3)
props = torch.cuda.get_device_properties(0)
print(f"GPU name={props.name}")
print(f"GPU memory GiB={props.total_memory / (1024 ** 3):.2f}")
'@
Invoke-VisiblePython -Arguments @('-c', $EnvironmentProbe)

$ManifestProbe = @'
import json, pathlib, sys
from benchmark_v2.e9.constants import MODEL_IDS
p = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = p.get("variants", [])
ids = tuple(row.get("model_id") for row in rows)
assert len(rows) == 6 and ids == tuple(MODEL_IDS), (ids, MODEL_IDS)
assert all(row.get("formal_training") is True and row.get("control_training") is False for row in rows)
assert all(row.get("loss_id") == "msmg_dwu_loss" for row in rows)
print("E9 variant manifest: PASS (6/6)")
'@
Invoke-VisiblePython -Arguments @('-c', $ManifestProbe, $Manifest)

Invoke-VisiblePython -Arguments @('-m', 'compileall', '-q', (Join-Path $ProjectRoot 'custom_models\src\benchmark_v2\e9'))
Invoke-VisiblePython -Arguments @('-c', 'import benchmark_v2.e9; print("benchmark_v2.e9 import: PASS")')

$Arguments = @('-m', 'benchmark_v2.e9', '--analysis-root', $AnalysisRoot, 'precheck')
if (-not [string]::IsNullOrWhiteSpace($Original26Xlsx)) {
    $Arguments += @('--original26-xlsx', (Resolve-Path -LiteralPath $Original26Xlsx).Path)
}
$Result = Invoke-GyxNativeProcess -FilePath $Python -Arguments $Arguments -WorkingDirectory $ProjectRoot
if (-not [string]::IsNullOrWhiteSpace([string]$Result.Stdout)) { Write-Host ([string]$Result.Stdout).TrimEnd() }
if (-not [string]::IsNullOrWhiteSpace([string]$Result.Stderr)) { Write-Host -ForegroundColor DarkYellow ([string]$Result.Stderr).TrimEnd() }
exit $Result.ExitCode
