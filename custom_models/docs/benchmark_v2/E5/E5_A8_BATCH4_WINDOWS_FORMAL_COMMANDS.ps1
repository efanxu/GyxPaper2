[CmdletBinding()]
param(
    [ValidateSet('StaticAudit', 'Preflight', 'Run', 'Readiness', 'QuarantineExisting', 'ClearStaleLock')]
    [string]$Action = 'StaticAudit',
    [string]$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$InputPath,
    [string]$TargetPath,
    [string]$SourceRevision,
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$Python = $PythonExecutable
$Gate = Join-Path $ProjectRoot 'scripts\st_mgprompt_a8_batch4_gate.py'
$SourceRoot = Join-Path $ProjectRoot 'custom_models\src'
$AuditRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\audit\st_mgprompt_a8_batch4'
$PreflightRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\preflight\st_mgprompt_a8_batch4'
$ReportPath = Join-Path $AuditRoot 'a8_batch4_preflight_summary.json'
$ReadinessPath = Join-Path $AuditRoot 'a8_batch4_readiness.json'
$ReferencePath = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\audit\e5_scope27\E5_A8_BATCH4_REFERENCE.json'
$ChildLogRoot = Join-Path $AuditRoot 'preflight'

if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ProjectRoot 'dataset\sdwpf_model_input_base.parquet' }
if ([string]::IsNullOrWhiteSpace($TargetPath)) { $TargetPath = Join-Path $ProjectRoot 'dataset\sdwpf_eval_target.parquet' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable does not exist: $Python" }
if (-not (Test-Path -LiteralPath $Gate -PathType Leaf)) { throw "A8 gate does not exist: $Gate" }
New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
New-Item -ItemType Directory -Path $PreflightRoot -Force | Out-Null
$env:PYTHONPATH = $SourceRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $ProjectRoot

function Invoke-A8Gate {
    param([string]$Label, [string[]]$Arguments, [int[]]$AllowedExitCodes = @(0), [string]$ReportPath)
    $stdoutPath = Join-Path $AuditRoot ($Label + '.stdout.log')
    $stderrPath = Join-Path $AuditRoot ($Label + '.stderr.log')
    $lines = @(& $Python $Gate @Arguments 2> $stderrPath)
    $code = $LASTEXITCODE
    $text = (($lines | ForEach-Object { [string]$_ }) -join [Environment]::NewLine)
    if (-not [string]::IsNullOrWhiteSpace($text)) { $text | Tee-Object -FilePath $stdoutPath -Append | ForEach-Object { Write-Host $_ } }
    if ($AllowedExitCodes -notcontains $code) { throw "$Label failed with exit code $code" }
    $json = $null
    if (-not [string]::IsNullOrWhiteSpace($ReportPath) -and (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        $json = Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } elseif (-not [string]::IsNullOrWhiteSpace($text)) {
        $json = $text | ConvertFrom-Json
    }
    [pscustomobject]@{ ExitCode = $code; Json = $json; Stdout = $stdoutPath; Stderr = $stderrPath }
}

switch ($Action) {
    'StaticAudit' {
        $contract = Invoke-A8Gate -Label 'validate-contract' -Arguments @('validate-contract') -AllowedExitCodes @(0, 74)
        $lock = Invoke-A8Gate -Label 'lock-status' -Arguments @('lock-status') -AllowedExitCodes @(0, 74)
        if ($null -eq $lock.Json -or [string]$lock.Json.status -ne 'ABSENT') { throw "A8 lock is not ABSENT: $($lock.Json.status)" }
        $freeze = Invoke-A8Gate -Label 'freeze-plan' -Arguments @('freeze-plan')
        $plan = Invoke-A8Gate -Label 'preflight-plan' -Arguments @('preflight-plan')
        $dry = Invoke-A8Gate -Label 'dry-run' -Arguments @('dry-run') -AllowedExitCodes @(0, 74)
        $ready = Invoke-A8Gate -Label 'readiness' -Arguments @('readiness', '--report-path', $ReadinessPath, '--reference-path', $ReferencePath) -AllowedExitCodes @(0, 4) -ReportPath $ReadinessPath
        Write-Host "A8 static status=$($ready.Json.status); no GPU preflight or training was started."
        if ($contract.ExitCode -ne 0) { exit 74 }
        if ($ready.Json.status -eq 'READY_A8_BATCH4_PREREQUISITE') { exit 0 }
        exit 4
    }
    'Preflight' {
        $revision = if ([string]::IsNullOrWhiteSpace($SourceRevision)) { (& git rev-parse HEAD).Trim() } else { $SourceRevision }
        $result = Invoke-A8Gate -Label 'preflight' -Arguments @('preflight', '--report-path', $ReportPath, '--child-log-root', $ChildLogRoot, '--source-revision', $revision) -ReportPath $ReportPath
        if ($null -eq $result.Json -or $result.Json.status -ne 'PASS') { throw 'A8 exact preflight did not pass.' }
    }
    'Run' {
        $args = @('run', '--input-path', $InputPath, '--target-path', $TargetPath, '--log-root', (Join-Path $AuditRoot 'formal'))
        if (-not [string]::IsNullOrWhiteSpace($SourceRevision)) { $args += @('--source-revision', $SourceRevision) }
        $null = Invoke-A8Gate -Label 'run' -Arguments $args
    }
    'Readiness' {
        $result = Invoke-A8Gate -Label 'readiness' -Arguments @('readiness', '--report-path', $ReadinessPath, '--reference-path', $ReferencePath) -AllowedExitCodes @(0, 4) -ReportPath $ReadinessPath
        if ($result.Json.status -ne 'READY_A8_BATCH4_PREREQUISITE') { exit 4 }
    }
    'QuarantineExisting' {
        $args = @('quarantine-existing', '--model', 'st_mgprompt_a8')
        if ($Apply) { $args += '--apply' }
        $null = Invoke-A8Gate -Label 'quarantine-existing' -Arguments $args -AllowedExitCodes @(0, 74)
    }
    'ClearStaleLock' {
        $status = Invoke-A8Gate -Label 'lock-status' -Arguments @('lock-status') -AllowedExitCodes @(0, 74)
        if ($null -eq $status.Json -or $status.Json.status -ne 'STALE') { throw 'Only STALE A8 locks may be cleared explicitly.' }
        $null = Invoke-A8Gate -Label 'clear-stale-lock' -Arguments @('clear-stale-lock')
    }
}
