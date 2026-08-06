[CmdletBinding()]
param(
    [ValidateSet('StaticAudit', 'PreflightPlan', 'Preflight', 'Run', 'Readiness', 'Aggregate', 'QuarantineExisting', 'ClearStaleLock')]
    [string]$Action = 'StaticAudit',
    [string]$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$InputPath,
    [string]$TargetPath,
    [string]$PreflightRoot,
    [string]$Model,
    [switch]$RequireComplete,
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$Gate = Join-Path $ProjectRoot 'scripts\original_batch4_scope26_gate.py'
$ResultRoot = Join-Path $ProjectRoot 'custom_models\results\benchmark_v2_uniform_bs4'
$SourceRoot = Join-Path $ProjectRoot 'custom_models\src'

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python executable does not exist: $PythonExecutable" }
if (-not (Test-Path -LiteralPath $Gate -PathType Leaf)) { throw "Original scope gate does not exist: $Gate" }
$oldPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $env:PYTHONPATH = $ProjectRoot + [IO.Path]::PathSeparator + $SourceRoot
} else {
    $env:PYTHONPATH = $ProjectRoot + [IO.Path]::PathSeparator + $SourceRoot + [IO.Path]::PathSeparator + $oldPythonPath
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $ProjectRoot

function Invoke-OriginalGate {
    param([string[]]$Arguments)
    & $PythonExecutable $Gate @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Action) {
    'StaticAudit' {
        Invoke-OriginalGate @('validate-manifest')
        Invoke-OriginalGate @('freeze')
        Invoke-OriginalGate @('preflight-plan')
        Invoke-OriginalGate @('static-audit', '--output-root', $ResultRoot)
    }
    'PreflightPlan' { Invoke-OriginalGate @('preflight-plan') }
    'Preflight' {
        $arguments = @('preflight')
        if ($PreflightRoot) { $arguments += @('--preflight-root', $PreflightRoot) }
        Invoke-OriginalGate $arguments
    }
    'Run' {
        $arguments = @('run')
        if ($InputPath) { $arguments += @('--input-path', $InputPath) }
        if ($TargetPath) { $arguments += @('--target-path', $TargetPath) }
        if ($PreflightRoot) { $arguments += @('--preflight-root', $PreflightRoot) }
        Invoke-OriginalGate $arguments
    }
    'Readiness' { Invoke-OriginalGate @('readiness', '--output-root', $ResultRoot) }
    'Aggregate' {
        $arguments = @('aggregate', '--output-root', $ResultRoot)
        if ($RequireComplete) { $arguments += '--require-complete' }
        Invoke-OriginalGate $arguments
    }
    'QuarantineExisting' {
        if ([string]::IsNullOrWhiteSpace($Model)) { throw 'Model is required for an explicit quarantine action.' }
        $arguments = @('quarantine-existing', '--model', $Model, '--output-root', $ResultRoot)
        if ($Apply) { $arguments += '--apply' }
        Invoke-OriginalGate $arguments
    }
    'ClearStaleLock' { Invoke-OriginalGate @('clear-stale-lock') }
}
