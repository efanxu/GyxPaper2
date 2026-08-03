[CmdletBinding()]
param(
    [ValidateSet('DryRun', 'PreflightPlan', 'Run', 'Readiness', 'Aggregate')]
    [string]$Action = 'DryRun',
    [string]$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$InputPath,
    [string]$TargetPath,
    [string]$SourceRevision,
    [switch]$RequireComplete
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$gatePath = Join-Path $repoRoot 'scripts\original_batch4_scope26_gate.py'
$resultRoot = Join-Path $repoRoot 'custom_models\results\benchmark_v2_uniform_bs4'
$sourceRoot = Join-Path $repoRoot 'custom_models\src'

$oldPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $env:PYTHONPATH = $sourceRoot
} else {
    $env:PYTHONPATH = $sourceRoot + [IO.Path]::PathSeparator + $oldPythonPath
}

$arguments = @($gatePath)
switch ($Action) {
    'DryRun' {
        $arguments += @('dry-run', '--output-root', $resultRoot)
    }
    'PreflightPlan' {
        $arguments += @('preflight-plan')
    }
    'Run' {
        $arguments += @('run')
        if ($InputPath) { $arguments += @('--input-path', $InputPath) }
        if ($TargetPath) { $arguments += @('--target-path', $TargetPath) }
        if ($SourceRevision) { $arguments += @('--source-revision', $SourceRevision) }
    }
    'Readiness' {
        $arguments += @('readiness', '--output-root', $resultRoot)
    }
    'Aggregate' {
        $arguments += @('aggregate', '--output-root', $resultRoot)
        if ($RequireComplete) { $arguments += '--require-complete' }
    }
}

& $PythonExecutable @arguments
exit $LASTEXITCODE
