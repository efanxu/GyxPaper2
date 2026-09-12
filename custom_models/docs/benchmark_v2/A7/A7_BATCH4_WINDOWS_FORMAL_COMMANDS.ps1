[CmdletBinding()]
param(
    [ValidateSet('StaticAudit', 'Preflight', 'Run', 'Readiness', 'QuarantineExisting', 'ClearStaleLock')]
    [string]$Action = 'StaticAudit',
    [string]$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$InputPath,
    [string]$TargetPath,
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$Python = $PythonExecutable
$Gate = Join-Path $ProjectRoot 'scripts\st_mgprompt_a7_batch4_gate.py'
$NativeRunner = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\WINDOWS_NATIVE_PROCESS_RUNNER.ps1'
$SourceRoot = Join-Path $ProjectRoot 'custom_models\src'
$AuditRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\audit\st_mgprompt_a7_batch4'
$PreflightRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\preflight\st_mgprompt_a7_batch4'
$ReportPath = Join-Path $AuditRoot 'a7_batch4_preflight_summary.json'
$ReadinessPath = Join-Path $AuditRoot 'a7_batch4_readiness.json'
$ReferencePath = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\audit\a7\A7_BATCH4_REFERENCE.json'
$ChildLogRoot = Join-Path $AuditRoot 'preflight'

if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ProjectRoot 'dataset\sdwpf_model_input_base.parquet' }
if ([string]::IsNullOrWhiteSpace($TargetPath)) { $TargetPath = Join-Path $ProjectRoot 'dataset\sdwpf_eval_target.parquet' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable does not exist: $Python" }
if (-not (Test-Path -LiteralPath $Gate -PathType Leaf)) { throw "A7 gate does not exist: $Gate" }
if (-not (Test-Path -LiteralPath $NativeRunner -PathType Leaf)) { throw "Windows native process runner does not exist: $NativeRunner" }
New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
New-Item -ItemType Directory -Path $PreflightRoot -Force | Out-Null
$oldPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $env:PYTHONPATH = $ProjectRoot + [IO.Path]::PathSeparator + $SourceRoot
} else {
    $env:PYTHONPATH = $ProjectRoot + [IO.Path]::PathSeparator + $SourceRoot + [IO.Path]::PathSeparator + $oldPythonPath
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $ProjectRoot
. $NativeRunner

function Invoke-A7Gate {
    param([string]$Label, [string[]]$Arguments, [int[]]$AllowedExitCodes = @(0), [string]$ReportPath)
    $result = Invoke-GyxPythonGate `
        -Python $Python `
        -Gate $Gate `
        -Arguments $Arguments `
        -WorkingDirectory $ProjectRoot `
        -LogRoot $AuditRoot `
        -Label $Label `
        -AllowedExitCodes $AllowedExitCodes `
        -ReportPath $ReportPath
    return $result
}
switch ($Action) {
    'StaticAudit' {
        $contract = Invoke-A7Gate -Label 'validate-contract' -Arguments @('validate-contract') -AllowedExitCodes @(0, 74)
        $lock = Invoke-A7Gate -Label 'lock-status' -Arguments @('lock-status') -AllowedExitCodes @(0, 74)
        if ($null -eq $lock.Json -or [string]$lock.Json.status -ne 'ABSENT') { throw "A7 lock must be ABSENT before launch; ACTIVE/MALFORMED fail closed and STALE requires explicit ClearStaleLock: $($lock.Json.status)" }
        $freeze = Invoke-A7Gate -Label 'freeze-plan' -Arguments @('freeze-plan')
        $plan = Invoke-A7Gate -Label 'preflight-plan' -Arguments @('preflight-plan')
        $dry = Invoke-A7Gate -Label 'dry-run' -Arguments @('dry-run') -AllowedExitCodes @(0, 74)
        $ready = Invoke-A7Gate -Label 'readiness' -Arguments @('readiness', '--report-path', $ReadinessPath, '--reference-path', $ReferencePath) -AllowedExitCodes @(0, 4) -ReportPath $ReadinessPath
        Write-Host "A7 static status=$($ready.Json.status); no GPU preflight or training was started."
        if ($contract.ExitCode -ne 0) { exit 74 }
        if ($ready.Json.status -eq 'READY') { exit 0 }
        exit 4
    }
    'Preflight' {
        $result = Invoke-A7Gate -Label 'preflight' -Arguments @('preflight', '--report-path', $ReportPath, '--child-log-root', $ChildLogRoot) -ReportPath $ReportPath
        if ($null -eq $result.Json -or $result.Json.status -ne 'PASS') { throw 'A7 exact preflight did not pass.' }
    }
    'Run' {
        $args = @('run', '--input-path', $InputPath, '--target-path', $TargetPath, '--log-root', (Join-Path $AuditRoot 'formal'))
        $result = Invoke-A7Gate -Label 'run' -Arguments $args
        if ($null -eq $result.Json -or $result.Json.status -ne 'COMPLETED' -or $result.Json.readiness.status -ne 'READY') { throw 'A7 run post-acceptance did not reach READY.' }
    }
    'Readiness' {
        $result = Invoke-A7Gate -Label 'readiness' -Arguments @('readiness', '--report-path', $ReadinessPath, '--reference-path', $ReferencePath) -AllowedExitCodes @(0, 4) -ReportPath $ReadinessPath
        if ($result.Json.status -ne 'READY') { exit 4 }
    }
    'QuarantineExisting' {
        $args = @('quarantine-existing')
        if ($Apply) { $args += '--apply' }
        $null = Invoke-A7Gate -Label 'quarantine-existing' -Arguments $args -AllowedExitCodes @(0, 74)
    }
    'ClearStaleLock' {
        $status = Invoke-A7Gate -Label 'lock-status' -Arguments @('lock-status') -AllowedExitCodes @(0, 74)
        if ($null -eq $status.Json -or $status.Json.status -ne 'STALE') { throw 'Only STALE A7 locks may be cleared explicitly.' }
        $null = Invoke-A7Gate -Label 'clear-stale-lock' -Arguments @('clear-stale-lock')
    }
}
