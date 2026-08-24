[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\PaperProject\GyxPaper2',
    [string]$Python = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$AnalysisRoot,
    [string]$PreflightRoot,
    [string]$FormalLogRoot,
    [string]$Original26Xlsx
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) { throw "Missing ProjectRoot: $ProjectRoot" }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($AnalysisRoot)) { $AnalysisRoot = Join-Path $ProjectRoot 'custom_models\results\benchmark_v2\msmg_dwu_transfer_seed2026' }
if ([string]::IsNullOrWhiteSpace($PreflightRoot)) { $PreflightRoot = Join-Path $ProjectRoot 'custom_models\logs\benchmark_v2\e9\preflight' }
if ([string]::IsNullOrWhiteSpace($FormalLogRoot)) { $FormalLogRoot = Join-Path $ProjectRoot 'custom_models\logs\benchmark_v2\e9\formal' }
New-Item -ItemType Directory -Path $FormalLogRoot -Force | Out-Null
$env:PYTHONPATH = Join-Path $ProjectRoot 'custom_models\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'
$RunScript = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\E9\E9_RUN_ALL_6_WINDOWS.ps1'
$WrapperLog = Join-Path $FormalLogRoot 'autoshutdown.log'
$Utf8 = New-Object System.Text.UTF8Encoding($false)

$Arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $RunScript,
    '-ProjectRoot', $ProjectRoot, '-Python', $Python,
    '-AnalysisRoot', $AnalysisRoot, '-PreflightRoot', $PreflightRoot, '-FormalLogRoot', $FormalLogRoot
)
if (-not [string]::IsNullOrWhiteSpace($Original26Xlsx)) { $Arguments += @('-Original26Xlsx', $Original26Xlsx) }

& powershell.exe @Arguments *> $WrapperLog
$RunCode = $LASTEXITCODE
[IO.File]::WriteAllText((Join-Path $FormalLogRoot 'autoshutdown_run_all_exit_code.txt'), ([string]$RunCode + "`r`n"), $Utf8)

$Copies = @(
    @((Join-Path $FormalLogRoot 'failed_models.csv'), (Join-Path $FormalLogRoot 'autoshutdown_failed_models.csv')),
    @((Join-Path $FormalLogRoot 'completed_models.csv'), (Join-Path $FormalLogRoot 'autoshutdown_completed_models.csv')),
    @((Join-Path $AnalysisRoot 'E9_TRANSFER_READINESS.json'), (Join-Path $FormalLogRoot 'autoshutdown_transfer_readiness.json')),
    @((Join-Path $AnalysisRoot 'E9_READINESS.json'), (Join-Path $FormalLogRoot 'autoshutdown_final_readiness.json'))
)
foreach ($Pair in $Copies) {
    if (Test-Path -LiteralPath $Pair[0] -PathType Leaf) { Copy-Item -LiteralPath $Pair[0] -Destination $Pair[1] -Force }
}

[IO.File]::AppendAllText($WrapperLog, "`r`nrun_all_exit_code=$RunCode`r`nshutdown=scheduled_after_saved_state`r`n", $Utf8)
shutdown.exe /s /t 0
exit $RunCode
