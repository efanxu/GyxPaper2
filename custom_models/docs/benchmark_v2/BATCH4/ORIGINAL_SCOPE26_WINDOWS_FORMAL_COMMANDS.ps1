[CmdletBinding()]
param(
    [ValidateSet('StaticAudit', 'Preflight', 'Run', 'Readiness', 'Aggregate', 'QuarantineExisting', 'ClearStaleLock')]
    [string]$Action = 'StaticAudit',
    [string]$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$InputPath,
    [string]$TargetPath,
    [string]$SourceRevision,
    [string]$PreflightRoot,
    [string]$Model,
    [switch]$RequireComplete,
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$Launcher = $MyInvocation.MyCommand.Path
$Python = $PythonExecutable
$Gate = Join-Path $ProjectRoot 'scripts\original_batch4_scope26_gate.py'
$ManifestPath = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\BATCH4\CURRENT_BATCH4_SCOPE26_MANIFEST.json'
$ResultRoot = Join-Path $ProjectRoot 'custom_models\results\benchmark_v2_uniform_bs4'
$ResultCsvPath = Join-Path $ResultRoot 'original_scope26_metrics.csv'
$SourceRoot = Join-Path $ProjectRoot 'custom_models\src'
$AuditRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\audit\original_scope26'
$LogRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\formal\original_scope26_windows_commands'
$ExpectedScope = 'benchmark_v2_batch4_scope26_seed2026'
$ExpectedTrainable = 24
$ExpectedEvaluateOnly = 2
$ExpectedTotal = 26
$Head = ''
$OriginHead = ''
$script:Head = ''
$script:OriginHead = ''
$script:LastValidation = $null
$script:LastFreeze = $null
$script:LastPreflight = $null
$script:LastReadiness = $null
$script:LastAggregate = $null

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $InputPath = Join-Path $ProjectRoot 'dataset\sdwpf_model_input_base.parquet'
}
if ([string]::IsNullOrWhiteSpace($TargetPath)) {
    $TargetPath = Join-Path $ProjectRoot 'dataset\sdwpf_eval_target.parquet'
}

if ([string]::IsNullOrWhiteSpace($PreflightRoot)) {
    $PreflightRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\preflight\original_scope26'
}

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable does not exist: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Windows formal command file does not exist: $Launcher"
}
if (-not (Test-Path -LiteralPath $Gate -PathType Leaf)) {
    throw "Original scope gate does not exist: $Gate"
}
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Original scope manifest does not exist: $ManifestPath"
}
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
    throw "InputPath does not exist: $InputPath"
}
if (-not (Test-Path -LiteralPath $TargetPath -PathType Leaf)) {
    throw "TargetPath does not exist: $TargetPath"
}

New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$oldPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $env:PYTHONPATH = $SourceRoot
} else {
    $env:PYTHONPATH = $SourceRoot + [IO.Path]::PathSeparator + $oldPythonPath
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath $ProjectRoot

function Invoke-Gate {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0)
    )

    $logPath = Join-Path $LogRoot ($Label + '.log')
    $lines = @(& $Python $Gate @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $textLines = @($lines | ForEach-Object { [string]$_ })
    $text = ($textLines -join [Environment]::NewLine)
    if (-not [string]::IsNullOrWhiteSpace($text)) {
        $text | Tee-Object -LiteralPath $logPath -Append | ForEach-Object { Write-Host $_ }
    }
    if ($AllowedExitCodes -notcontains $exitCode) {
        throw "$Label failed with exit code $exitCode. See $logPath"
    }
    $json = $null
    if (-not [string]::IsNullOrWhiteSpace($text)) {
        try {
            $json = $text | ConvertFrom-Json
        } catch {
            $json = $null
        }
    }
    return [pscustomobject]@{
        Label = $Label
        ExitCode = $exitCode
        Text = $text
        Json = $json
        LogPath = $logPath
    }
}

function Invoke-GitText {
    param([string[]]$Arguments)
    $result = @(& git -C $ProjectRoot @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $exitCode."
    }
    return (($result | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
}

function Assert-RepositoryIdentity {
    $branch = Invoke-GitText @('branch', '--show-current')
    if ($branch -ne 'main') {
        throw "Formal Original scope must run on branch main; current branch is $branch."
    }
    $head = Invoke-GitText @('rev-parse', 'HEAD')
    $originHead = Invoke-GitText @('rev-parse', 'origin/main')
    $script:Head = $head
    $script:OriginHead = $originHead
    if ($head -ne $originHead) {
        throw "HEAD does not equal origin/main: HEAD=$head origin/main=$originHead"
    }
    $trackedChanges = Invoke-GitText @('status', '--porcelain', '--untracked-files=no')
    if (-not [string]::IsNullOrWhiteSpace($trackedChanges)) {
        throw "Tracked worktree changes are present; commit or resolve them before formal execution."
    }
    Write-Host "Repository identity: branch=$branch HEAD=$head origin/main=$originHead"
}

function Assert-StaticPlan {
    $validation = Invoke-Gate -Label 'validate-manifest' -Arguments @('validate-manifest')
    if ($null -eq $validation.Json -or $validation.Json.scope_id -ne $ExpectedScope) {
        throw 'Manifest validation did not return the active Original scope26 identity.'
    }

    $freeze = Invoke-Gate -Label 'freeze' -Arguments @('freeze')
    if ($null -eq $freeze.Json -or [string]::IsNullOrWhiteSpace([string]$freeze.Json.freeze_hash)) {
        throw 'Original freeze did not produce a freeze_hash.'
    }

    $preflightPlan = Invoke-Gate -Label 'preflight-plan' -Arguments @('preflight-plan')
    if ($null -eq $preflightPlan.Json) {
        throw 'Original preflight-plan did not return JSON.'
    }
    if ([int]$preflightPlan.Json.counts.trainable_expected -ne $ExpectedTrainable -or
        [int]$preflightPlan.Json.counts.evaluate_only_expected -ne $ExpectedEvaluateOnly) {
        throw 'Original exact preflight-plan denominator is not 24 trainable plus 2 evaluate-only.'
    }

    $dryRun = Invoke-Gate -Label 'dry-run' -Arguments @('dry-run', '--output-root', $ResultRoot)
    if ($null -eq $dryRun.Json -or @($dryRun.Json.entries).Count -ne $ExpectedTotal) {
        throw 'Original dry-run did not return all 26 active entries.'
    }
    $blocked = @($dryRun.Json.entries | Where-Object { [string]$_.action -like 'BLOCK*' })
    foreach ($row in $blocked) {
        $model = [string]$row.model_id
        Write-Host "MANUAL REVIEW ONLY: $model action=$($row.action) reasons=$([string]::Join(';', @($row.reasons)))"
        Write-Host "Preview only: & '$Launcher' -Action QuarantineExisting -Model '$model'"
        Write-Host "After independent per-model confirmation only: & '$Launcher' -Action QuarantineExisting -Model '$model' -Apply"
    }

    $lock = Invoke-Gate -Label 'lock-status' -Arguments @('lock-status') -AllowedExitCodes @(0, 73, 74)
    if ($null -eq $lock.Json) {
        throw 'Original lock-status did not return JSON.'
    }
    if ([string]$lock.Json.status -ne 'ABSENT') {
        if ([string]$lock.Json.status -eq 'STALE') {
            Write-Host "Manual action after independent inspection: & '$Launcher' -Action ClearStaleLock"
        }
        throw "Formal execution is blocked by lock status $($lock.Json.status); no automatic lock action is permitted."
    }

    $readiness = Invoke-Gate -Label 'readiness' -Arguments @('readiness', '--output-root', $ResultRoot) -AllowedExitCodes @(0, 4)
    if ($null -eq $readiness.Json) {
        throw 'Original readiness did not return JSON.'
    }
    if ([string]$readiness.Json.scope_id -ne $ExpectedScope) {
        throw 'Original readiness returned the wrong scope identity.'
    }
    $script:LastValidation = $validation.Json
    $script:LastFreeze = $freeze.Json
    $script:LastReadiness = $readiness.Json
    Write-Host "Static readiness status: $($readiness.Json.status)"
    return [pscustomobject]@{
        Validation = $validation.Json
        Freeze = $freeze.Json
        PreflightPlan = $preflightPlan.Json
        DryRun = $dryRun.Json
        Readiness = $readiness.Json
    }
}

function Write-FinalEvidence {
    $manifestHash = ''
    $runMapHash = ''
    $freezeHash = ''
    $readinessStatus = 'NOT_RUN'
    $aggregateStatus = 'NOT_RUN'
    if ($null -ne $script:LastValidation) {
        $manifestHash = [string]$script:LastValidation.manifest_hash
        $runMapHash = [string]$script:LastValidation.run_map_hash
    }
    if ($null -ne $script:LastFreeze) {
        $freezeHash = [string]$script:LastFreeze.freeze_hash
    }
    if ($null -ne $script:LastReadiness) {
        $readinessStatus = [string]$script:LastReadiness.status
    }
    if ($null -ne $script:LastAggregate) {
        $aggregateStatus = [string]$script:LastAggregate.status
    }
    Write-Host "FINAL EVIDENCE HEAD=$($script:Head)"
    Write-Host "FINAL EVIDENCE origin/main=$($script:OriginHead)"
    Write-Host "FINAL EVIDENCE manifest_hash=$manifestHash"
    Write-Host "FINAL EVIDENCE run_map_hash=$runMapHash"
    Write-Host "FINAL EVIDENCE freeze_hash=$freezeHash"
    Write-Host "FINAL EVIDENCE readiness_status=$readinessStatus"
    Write-Host "FINAL EVIDENCE aggregate_status=$aggregateStatus"
    Write-Host "FINAL EVIDENCE result_csv=$ResultCsvPath"
    Write-Host "FINAL EVIDENCE log_root=$LogRoot"
}

function Invoke-ExactPreflight {
    $null = Assert-RepositoryIdentity
    $null = Assert-StaticPlan
    $revision = $script:Head
    if (-not [string]::IsNullOrWhiteSpace($SourceRevision)) {
        $revision = $SourceRevision
    }
    $arguments = @('preflight', '--preflight-root', $PreflightRoot, '--source-revision', $revision)
    $result = Invoke-Gate -Label 'preflight' -Arguments $arguments
    if ($null -eq $result.Json -or $result.Json.gpu_preflight_performed -ne $true) {
        throw 'Exact Original GPU preflight did not report gpu_preflight_performed=true.'
    }
    if ([int]$result.Json.counts.trainable -ne $ExpectedTrainable -or
        [int]$result.Json.counts.expected_trainable -ne $ExpectedTrainable -or
        @($result.Json.results | Where-Object { [string]$_.status -ne 'PASS' }).Count -ne 0) {
        throw 'Exact Original GPU preflight did not pass 24/24 trainable entries.'
    }
    $script:LastPreflight = $result.Json
    Write-Host "Exact preflight exit_code=$($result.ExitCode) log=$($result.LogPath)"
    Write-Host 'Exact Original GPU preflight PASS: 24/24 trainable; 2 evaluate-only skipped.'
}

function Invoke-FormalRun {
    if ([string]::IsNullOrWhiteSpace($InputPath) -or [string]::IsNullOrWhiteSpace($TargetPath)) {
        throw 'Run requires -InputPath and -TargetPath.'
    }
    if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
        throw "InputPath does not exist: $InputPath"
    }
    if (-not (Test-Path -LiteralPath $TargetPath -PathType Leaf)) {
        throw "TargetPath does not exist: $TargetPath"
    }
    $null = Assert-RepositoryIdentity
    $null = Assert-StaticPlan
    if (-not (Test-Path -LiteralPath $PreflightRoot -PathType Container)) {
        throw "PreflightRoot does not exist; run the explicit Preflight action first: $PreflightRoot"
    }
    $revision = $script:Head
    if (-not [string]::IsNullOrWhiteSpace($SourceRevision)) {
        $revision = $SourceRevision
    }
    $arguments = @('run', '--input-path', $InputPath, '--target-path', $TargetPath, '--preflight-root', $PreflightRoot, '--source-revision', $revision)
    $result = Invoke-Gate -Label 'run' -Arguments $arguments
    if ($null -eq $result.Json) {
        throw 'Formal Original run did not return JSON.'
    }
    Write-Host "Formal run exit_code=$($result.ExitCode) log=$($result.LogPath) per-model logs=$LogRoot"
    Write-Host 'Formal Original run completed; readiness remains the authoritative denominator gate.'
}

function Invoke-ReadinessOnly {
    $null = Assert-RepositoryIdentity
    $result = Invoke-Gate -Label 'readiness-only' -Arguments @('readiness', '--output-root', $ResultRoot) -AllowedExitCodes @(0, 4)
    if ($null -eq $result.Json) {
        throw 'Original readiness did not return JSON.'
    }
    $script:LastReadiness = $result.Json
    Write-Host "Readiness exit_code=$($result.ExitCode) log=$($result.LogPath)"
    if ([string]$result.Json.status -eq 'COMPLETED_READY_26_OF_26') {
        Write-Host 'Original readiness PASS: COMPLETED_READY_26_OF_26.'
    } else {
        Write-Host "Original readiness is not complete: $($result.Json.status)"
    }
}

function Invoke-AggregateOnly {
    if (-not $RequireComplete) {
        throw 'Aggregate requires the explicit -RequireComplete switch.'
    }
    $null = Assert-RepositoryIdentity
    $readiness = Invoke-Gate -Label 'aggregate-readiness' -Arguments @('readiness', '--output-root', $ResultRoot) -AllowedExitCodes @(0, 4)
    if ($null -eq $readiness.Json -or [string]$readiness.Json.status -ne 'COMPLETED_READY_26_OF_26') {
        throw 'Aggregate is refused unless readiness is exactly COMPLETED_READY_26_OF_26.'
    }
    $script:LastReadiness = $readiness.Json
    $result = Invoke-Gate -Label 'aggregate' -Arguments @('aggregate', '--output-root', $ResultRoot, '--require-complete')
    $script:LastAggregate = $result.Json
    Write-Host "Aggregate exit_code=$($result.ExitCode) log=$($result.LogPath)"
}

function Invoke-QuarantineExisting {
    if ([string]::IsNullOrWhiteSpace($Model)) {
        throw 'QuarantineExisting requires -Model and operates on one model at a time.'
    }
    $null = Assert-RepositoryIdentity
    $arguments = @('quarantine-existing', '--output-root', $ResultRoot, '--model', $Model)
    if ($Apply) {
        Write-Host "EXPLICIT APPLY requested for one model only: $Model"
        $arguments += '--apply'
    } else {
        Write-Host "PREVIEW ONLY for $Model; no result directory will be moved."
    }
    $null = Invoke-Gate -Label ('quarantine-' + $Model) -Arguments $arguments
}

function Invoke-ClearStaleLock {
    $null = Assert-RepositoryIdentity
    $lock = Invoke-Gate -Label 'lock-status-before-clear' -Arguments @('lock-status') -AllowedExitCodes @(0, 73, 74)
    if ($null -eq $lock.Json -or [string]$lock.Json.status -ne 'STALE') {
        throw "ClearStaleLock refuses any lock status other than STALE: $($lock.Json.status)"
    }
    $null = Invoke-Gate -Label 'clear-stale-lock' -Arguments @('clear-stale-lock')
}

switch ($Action) {
    'StaticAudit' {
        $null = Assert-RepositoryIdentity
        $null = Assert-StaticPlan
        Write-FinalEvidence
        Write-Host 'Static Original scope26 audit complete. No GPU preflight, training, evaluation, aggregate, cleanup, quarantine, lock clearing, or shutdown was started.'
    }
    'Preflight' {
        Invoke-ExactPreflight
        Write-FinalEvidence
    }
    'Run' {
        Invoke-FormalRun
        Write-FinalEvidence
    }
    'Readiness' {
        Invoke-ReadinessOnly
        Write-FinalEvidence
    }
    'Aggregate' {
        Invoke-AggregateOnly
        Write-FinalEvidence
    }
    'QuarantineExisting' {
        Invoke-QuarantineExisting
    }
    'ClearStaleLock' {
        Invoke-ClearStaleLock
    }
}
