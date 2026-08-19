[CmdletBinding()]
param(
    [ValidateSet('StaticAudit', 'Preflight', 'Run', 'Readiness', 'Aggregate', 'ClearStaleLock')]
    [string]$Action = 'StaticAudit',
    [string]$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$InputPath,
    [string]$TargetPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$Python = $PythonExecutable
$Gate = Join-Path $ProjectRoot 'scripts\e5_batch4_scope27_gate.py'
$A8Gate = Join-Path $ProjectRoot 'scripts\st_mgprompt_a8_batch4_gate.py'
$Manifest = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\E5\E5_SCOPE27_VARIANT_MANIFEST.json'
$Requirements = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\E5\requirements.txt'
$NativeRunner = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\WINDOWS_NATIVE_PROCESS_RUNNER.ps1'
$SourceRoot = Join-Path $ProjectRoot 'custom_models\src'
$OutputRoot = Join-Path $ProjectRoot 'custom_models\results\benchmark_v2_uniform_bs4\common_loss_architecture_seed2026'
$AuditRoot = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\audit\e5_scope27'
$PreflightRoot = Join-Path $AuditRoot 'preflight'
$ChildLogRoot = Join-Path $PreflightRoot 'child_logs'
$PreflightReport = Join-Path $AuditRoot 'e5_scope27_preflight_summary.json'
$SuiteReport = Join-Path $AuditRoot 'e5_scope27_suite_report.json'
$ReadinessReport = Join-Path $AuditRoot 'e5_scope27_readiness.json'
$EvidencePath = Join-Path $AuditRoot 'e5_scope27_evidence.json'
$A8ReadinessReport = Join-Path $ProjectRoot 'custom_models\logs\uniform_bs4\audit\st_mgprompt_a8_batch4\a8_batch4_readiness.json'
$A8ReferencePath = Join-Path $AuditRoot 'E5_A8_BATCH4_REFERENCE.json'

if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ProjectRoot 'dataset\sdwpf_model_input_base.parquet' }
if ([string]::IsNullOrWhiteSpace($TargetPath)) { $TargetPath = Join-Path $ProjectRoot 'dataset\sdwpf_eval_target.parquet' }
foreach ($requiredFile in @($Python, $Gate, $A8Gate, $Manifest, $Requirements, $NativeRunner)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "Required file does not exist: $requiredFile" }
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
New-Item -ItemType Directory -Path $PreflightRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ChildLogRoot -Force | Out-Null

$oldPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $env:PYTHONPATH = $ProjectRoot + [IO.Path]::PathSeparator + $SourceRoot
} else {
    $env:PYTHONPATH = $ProjectRoot + [IO.Path]::PathSeparator + $SourceRoot + [IO.Path]::PathSeparator + $oldPythonPath
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
Set-Location -LiteralPath $ProjectRoot
. $NativeRunner

function Invoke-E5Gate {
    param([string]$Label, [string[]]$Arguments, [int[]]$AllowedExitCodes = @(0), [string]$ReportPath)
    $gateArguments = @('--manifest', $Manifest) + @($Arguments)
    $result = Invoke-GyxPythonGate `
        -Python $Python `
        -Gate $Gate `
        -Arguments $gateArguments `
        -WorkingDirectory $ProjectRoot `
        -LogRoot $AuditRoot `
        -Label $Label `
        -AllowedExitCodes $AllowedExitCodes `
        -ReportPath $ReportPath
    return $result
}

function Invoke-A8Readiness {
    $result = Invoke-GyxPythonGate `
        -Python $Python `
        -Gate $A8Gate `
        -Arguments @('readiness', '--report-path', $A8ReadinessReport, '--reference-path', $A8ReferencePath) `
        -WorkingDirectory $ProjectRoot `
        -LogRoot $AuditRoot `
        -Label 'a8-readiness' `
        -AllowedExitCodes @(0, 4) `
        -ReportPath $A8ReadinessReport
    return $result
}

function Assert-A8Ready {
    $result = Invoke-A8Readiness
    if ($null -eq $result.Json -or [string]$result.Json.status -ne 'READY') {
        throw 'A8 prerequisite is NOT_READY. Complete the A8 Windows launcher through Readiness before E5.'
    }
}

function Assert-TrainingInputs {
    foreach ($path in @($InputPath, $TargetPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Formal dataset file does not exist: $path" }
    }
}

switch ($Action) {
    'StaticAudit' {
        $dependency = Invoke-GyxNativeProcess -FilePath $Python -Arguments @('-c', 'import openpyxl; print(openpyxl.__version__)') -WorkingDirectory $ProjectRoot
        if ($dependency.ExitCode -ne 0) {
            throw "E5 dependency check failed. Run: & '$Python' -m pip install -r '$Requirements'"
        }
        $manifestResult = Invoke-E5Gate -Label 'validate-manifest' -Arguments @('validate-manifest')
        if ($null -eq $manifestResult.Json -or [string]$manifestResult.Json.status -ne 'PASS') { throw 'E5 manifest validation did not pass.' }
        $lock = Invoke-E5Gate -Label 'lock-status' -Arguments @('lock-status') -AllowedExitCodes @(0, 74)
        if ($null -eq $lock.Json -or [string]$lock.Json.status -ne 'ABSENT') { throw "E5 lock must be ABSENT before launch; ACTIVE/MALFORMED fail closed and STALE requires explicit ClearStaleLock: $($lock.Json.status)" }
        $freeze = Invoke-E5Gate -Label 'freeze-plan' -Arguments @('freeze-plan')
        $plan = Invoke-E5Gate -Label 'preflight-plan' -Arguments @('preflight-plan')
        if ($null -eq $plan.Json -or [int]$plan.Json.required -ne 24 -or @($plan.Json.entries).Count -ne 24) { throw 'E5 preflight plan is not the frozen 24-model training scope.' }
        $dry = Invoke-E5Gate -Label 'dry-run' -Arguments @('dry-run', '--output-root', $OutputRoot)
        $a8 = Invoke-A8Readiness
        Write-Host "E5 static audit PASS; precision=FP32; amp_enabled=false; A8 status=$($a8.Json.status). No GPU preflight or training was started."
        if ($null -ne $a8.Json -and [string]$a8.Json.status -eq 'READY') { exit 0 }
        exit 4
    }
    'Preflight' {
        Assert-A8Ready
        $result = Invoke-E5Gate -Label 'preflight' -Arguments @(
            'preflight', '--preflight-root', $PreflightRoot,
            '--report-path', $PreflightReport, '--child-log-root', $ChildLogRoot
        ) -ReportPath $PreflightReport
        if ($null -eq $result.Json -or [string]$result.Json.status -ne 'PASS' -or
            [int]$result.Json.counts.pass -ne 24 -or [int]$result.Json.counts.expected -ne 24) {
            throw 'E5 exact FP32 preflight did not pass for all 24 trainable models.'
        }
    }
    'Run' {
        Assert-TrainingInputs
        Assert-A8Ready
        $result = Invoke-E5Gate -Label 'run' -Arguments @(
            'run', '--input-path', $InputPath, '--target-path', $TargetPath,
            '--log-root', (Join-Path $AuditRoot 'runs'), '--preflight-root', $PreflightRoot,
            '--suite-report-path', $SuiteReport
        ) -ReportPath $SuiteReport
        if ($null -eq $result.Json -or [string]$result.Json.status -ne 'COMPLETED' -or
            [int]$result.Json.entry_count -ne 27 -or [string]$result.Json.readiness.status -ne 'READY') {
            throw 'E5 formal run did not complete and reach READY for all 27 scope entries.'
        }
    }
    'Readiness' {
        Assert-A8Ready
        $result = Invoke-E5Gate -Label 'readiness' -Arguments @(
            'readiness', '--output-root', $OutputRoot,
            '--report-path', $ReadinessReport, '--evidence-path', $EvidencePath
        ) -AllowedExitCodes @(0, 4) -ReportPath $ReadinessReport
        if ($null -eq $result.Json -or [string]$result.Json.status -ne 'READY') { exit 4 }
    }
    'Aggregate' {
        Assert-A8Ready
        $result = Invoke-E5Gate -Label 'aggregate' -Arguments @(
            'aggregate', '--output-root', $OutputRoot,
            '--report-path', $ReadinessReport, '--evidence-path', $EvidencePath,
            '--require-complete'
        )
        if ($null -eq $result.Json) { throw 'E5 aggregation returned no JSON result.' }
    }
    'ClearStaleLock' {
        $status = Invoke-E5Gate -Label 'lock-status' -Arguments @('lock-status') -AllowedExitCodes @(0, 74)
        if ($null -eq $status.Json -or [string]$status.Json.status -ne 'STALE') { throw 'Only STALE E5 locks may be cleared explicitly.' }
        $null = Invoke-E5Gate -Label 'clear-stale-lock' -Arguments @('clear-stale-lock')
    }
}
exit 0
