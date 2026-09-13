[CmdletBinding()]
param(
    [string]$PythonExecutable = 'python'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$SourceRoot = Join-Path $ProjectRoot 'custom_models\src'
$PythonCommand = Get-Command $PythonExecutable -ErrorAction Stop
$Python = $PythonCommand.Source

$previousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
    $env:PYTHONPATH = $SourceRoot
} else {
    $env:PYTHONPATH = $SourceRoot + [IO.Path]::PathSeparator + $previousPythonPath
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

Set-Location -LiteralPath $ProjectRoot
& $Python -m st_mgprompt.run_ablation --variants A6 --run-full
if ($LASTEXITCODE -ne 0) {
    throw "A6 Single-direction Cross Fusion Full run failed with exit code $LASTEXITCODE."
}
