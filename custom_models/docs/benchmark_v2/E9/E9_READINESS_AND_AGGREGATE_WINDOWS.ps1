[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\PaperProject\GyxPaper2',
    [string]$Python = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [Parameter(Mandatory = $true)]
    [string]$Original26Xlsx,
    [string]$AnalysisRoot,
    [string]$ArtifactNode,
    [string]$ArtifactNodeModules
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) { throw "Missing ProjectRoot: $ProjectRoot" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
if (-not (Test-Path -LiteralPath $Original26Xlsx -PathType Leaf)) { throw "Missing original26 workbook: $Original26Xlsx" }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
$Original26Xlsx = (Resolve-Path -LiteralPath $Original26Xlsx).Path
if ([string]::IsNullOrWhiteSpace($AnalysisRoot)) { $AnalysisRoot = Join-Path $ProjectRoot 'custom_models\results\benchmark_v2\msmg_dwu_transfer_seed2026' }

$env:PYTHONPATH = Join-Path $ProjectRoot 'custom_models\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'
if (-not [string]::IsNullOrWhiteSpace($ArtifactNode)) { $env:E9_ARTIFACT_NODE = $ArtifactNode }
if (-not [string]::IsNullOrWhiteSpace($ArtifactNodeModules)) { $env:E9_ARTIFACT_NODE_MODULES = $ArtifactNodeModules }
Set-Location -LiteralPath $ProjectRoot

& $Python -m benchmark_v2.e9 --analysis-root $AnalysisRoot readiness --original26-xlsx $Original26Xlsx
$ReadinessCode = $LASTEXITCODE
$ReadinessPath = Join-Path $AnalysisRoot 'E9_READINESS.json'
if (-not (Test-Path -LiteralPath $ReadinessPath -PathType Leaf)) { throw "Readiness file was not created: $ReadinessPath" }
$Readiness = Get-Content -LiteralPath $ReadinessPath -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "CORE_E9_READY=$($Readiness.CORE_E9_READY)"
Write-Host "LOSS_ONLY_PAIRING_READY=$($Readiness.LOSS_ONLY_PAIRING_READY)"
if ($ReadinessCode -ne 0 -or $Readiness.CORE_E9_READY -ne '12/12' -or $Readiness.LOSS_ONLY_PAIRING_READY -ne '6/6') {
    Write-Host 'Incomplete E9 pairs:'
    $Readiness.pairs | Where-Object { $_.pair_status -ne 'READY' } | Select-Object model_id, control_ready, transfer_ready, loss_only_diff_valid, pair_status | Format-Table -AutoSize
    exit 2
}

if ([string]::IsNullOrWhiteSpace($env:E9_ARTIFACT_NODE) -or -not (Test-Path -LiteralPath $env:E9_ARTIFACT_NODE -PathType Leaf)) {
    throw 'Aggregate requires E9_ARTIFACT_NODE or -ArtifactNode pointing to node.exe.'
}
if ([string]::IsNullOrWhiteSpace($env:E9_ARTIFACT_NODE_MODULES) -or -not (Test-Path -LiteralPath $env:E9_ARTIFACT_NODE_MODULES -PathType Container)) {
    throw 'Aggregate requires E9_ARTIFACT_NODE_MODULES or -ArtifactNodeModules pointing to the artifact-tool node_modules directory.'
}

& $Python -m benchmark_v2.e9 --analysis-root $AnalysisRoot aggregate --original26-xlsx $Original26Xlsx --require-complete
exit $LASTEXITCODE
