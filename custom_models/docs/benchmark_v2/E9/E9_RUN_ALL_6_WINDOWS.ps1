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
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
if ([string]::IsNullOrWhiteSpace($AnalysisRoot)) { $AnalysisRoot = Join-Path $ProjectRoot 'custom_models\results\benchmark_v2\msmg_dwu_transfer_seed2026' }
if ([string]::IsNullOrWhiteSpace($PreflightRoot)) { $PreflightRoot = Join-Path $ProjectRoot 'custom_models\logs\benchmark_v2\e9\preflight' }
if ([string]::IsNullOrWhiteSpace($FormalLogRoot)) { $FormalLogRoot = Join-Path $ProjectRoot 'custom_models\logs\benchmark_v2\e9\formal' }
$ManifestPath = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\E9\E9_VARIANT_MANIFEST.json'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "Missing E9 variant manifest: $ManifestPath" }
if (-not [string]::IsNullOrWhiteSpace($Original26Xlsx) -and -not (Test-Path -LiteralPath $Original26Xlsx -PathType Leaf)) {
    throw "Missing original26 workbook: $Original26Xlsx"
}

New-Item -ItemType Directory -Path $AnalysisRoot -Force | Out-Null
New-Item -ItemType Directory -Path $FormalLogRoot -Force | Out-Null
$MainLog = Join-Path $FormalLogRoot 'run_all_6.log'
$FailedPath = Join-Path $FormalLogRoot 'failed_models.csv'
$CompletedPath = Join-Path $FormalLogRoot 'completed_models.csv'
$ExitPath = Join-Path $FormalLogRoot 'run_all_exit_code.txt'
$PidPath = Join-Path $FormalLogRoot 'run_all.pid'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($FailedPath, "model_id,run_id,exit_code`r`n", $Utf8)
[IO.File]::WriteAllText($CompletedPath, "model_id,run_id,status`r`n", $Utf8)
[IO.File]::WriteAllText($PidPath, ([string]$PID + "`r`n"), $Utf8)

$env:PYTHONPATH = Join-Path $ProjectRoot 'custom_models\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath $ProjectRoot
. (Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\WINDOWS_NATIVE_PROCESS_RUNNER.ps1')

function Write-MainLog {
    param([string]$Text)
    $Line = ('{0} {1}' -f ([DateTime]::Now.ToString('s')), $Text)
    [IO.File]::AppendAllText($MainLog, $Line + "`r`n", $Utf8)
    Write-Host $Line
}

function Save-ProcessLog {
    param([string]$Path, $Native)
    $Content = [string]$Native.Stdout
    if (-not [string]::IsNullOrWhiteSpace([string]$Native.Stderr)) {
        $Content += "`r`n[stderr]`r`n" + [string]$Native.Stderr
    }
    [IO.File]::WriteAllText($Path, $Content, $Utf8)
}

$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (@($Manifest.variants).Count -ne 6) { throw 'E9 variant manifest must contain exactly six rows.' }
$Overall = 0
Write-MainLog ('START run-all HEAD=' + (& git rev-parse HEAD))
Write-MainLog "Python=$Python"

foreach ($Variant in @($Manifest.variants)) {
    if ($Variant.formal_training -ne $true -or $Variant.control_training -ne $false -or $Variant.loss_id -ne 'msmg_dwu_loss') {
        throw "Invalid E9 transfer row in manifest: $($Variant.variant_id)"
    }
    $ModelId = [string]$Variant.model_id
    $RunId = [string]$Variant.transfer_run_id
    $ModelLog = Join-Path $FormalLogRoot ($RunId + '.log')
    Write-MainLog "START model=$ModelId run_id=$RunId"
    $Arguments = @(
        '-m', 'benchmark_v2.e9', '--analysis-root', $AnalysisRoot,
        'train-one', '--model', $ModelId, '--device', 'cuda', '--preflight-root', $PreflightRoot
    )
    $ModelStderrLog = Join-Path $FormalLogRoot ($RunId + '.stderr.log')
    try {
        $Process = Start-Process `
            -FilePath $Python `
            -ArgumentList (Join-GyxWindowsCommandLine -Arguments $Arguments) `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $ModelLog `
            -RedirectStandardError $ModelStderrLog `
            -WindowStyle Hidden `
            -PassThru `
            -Wait
        $Code = [int]$Process.ExitCode
        if ((Test-Path -LiteralPath $ModelStderrLog -PathType Leaf) -and (Get-Item -LiteralPath $ModelStderrLog).Length -gt 0) {
            [IO.File]::AppendAllText($ModelLog, "`r`n[stderr]`r`n" + (Get-Content -LiteralPath $ModelStderrLog -Raw -Encoding UTF8), $Utf8)
        }
    } catch {
        $Code = 97
        [IO.File]::WriteAllText($ModelLog, "PROCESS_LAUNCH_FAILURE`r`n$($_.Exception.Message)`r`n", $Utf8)
    }
    if ($Code -eq 0) {
        [IO.File]::AppendAllText($CompletedPath, "$ModelId,$RunId,COMPLETED_OR_SAFE_SKIP`r`n", $Utf8)
    } else {
        [IO.File]::AppendAllText($FailedPath, "$ModelId,$RunId,$Code`r`n", $Utf8)
        $Overall = 1
    }
    Write-MainLog "END model=$ModelId run_id=$RunId exit=$Code log=$ModelLog"
}

$TransferArguments = @('-m', 'benchmark_v2.e9', '--analysis-root', $AnalysisRoot, 'transfer-readiness')
$TransferResult = Invoke-GyxNativeProcess -FilePath $Python -Arguments $TransferArguments -WorkingDirectory $ProjectRoot
Save-ProcessLog -Path (Join-Path $FormalLogRoot 'transfer_readiness.log') -Native $TransferResult
Write-MainLog "transfer-readiness exit=$($TransferResult.ExitCode)"
if ($TransferResult.ExitCode -ne 0) { $Overall = 1 }

if (-not [string]::IsNullOrWhiteSpace($Original26Xlsx)) {
    $ResolvedWorkbook = (Resolve-Path -LiteralPath $Original26Xlsx).Path
    $ReadinessArguments = @('-m', 'benchmark_v2.e9', '--analysis-root', $AnalysisRoot, 'readiness', '--original26-xlsx', $ResolvedWorkbook)
    $ReadinessResult = Invoke-GyxNativeProcess -FilePath $Python -Arguments $ReadinessArguments -WorkingDirectory $ProjectRoot
    Save-ProcessLog -Path (Join-Path $FormalLogRoot 'full_readiness.log') -Native $ReadinessResult
    Write-MainLog "full-readiness exit=$($ReadinessResult.ExitCode)"
    if ($ReadinessResult.ExitCode -ne 0) { $Overall = 1 }
}

[IO.File]::WriteAllText($ExitPath, ([string]$Overall + "`r`n"), $Utf8)
Write-MainLog "END run-all exit=$Overall completed=$CompletedPath failed=$FailedPath"
exit $Overall
