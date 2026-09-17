[CmdletBinding()]
param(
    [ValidateSet('A6-C1', 'A6-C2', 'A6-C3', 'All')]
    [string]$Candidate = 'All',
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

$Variants = if ($Candidate -eq 'All') {
    @('A6-C1', 'A6-C2', 'A6-C3')
} else {
    @($Candidate)
}

Set-Location -LiteralPath $ProjectRoot
& $Python -m st_mgprompt.run_ablation --variants $Variants --run-full
if ($LASTEXITCODE -ne 0) {
    throw "Cross-Fusion candidate Full run failed with exit code $LASTEXITCODE."
}
