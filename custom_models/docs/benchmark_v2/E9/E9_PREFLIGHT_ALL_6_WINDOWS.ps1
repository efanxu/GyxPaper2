[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\PaperProject\GyxPaper2',
    [string]$Python = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',
    [string]$PreflightRoot
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) { throw "Missing ProjectRoot: $ProjectRoot" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
if ([string]::IsNullOrWhiteSpace($PreflightRoot)) {
    $PreflightRoot = Join-Path $ProjectRoot 'custom_models\logs\benchmark_v2\e9\preflight'
}
$InventoryPath = Join-Path $PreflightRoot 'E9_PREFLIGHT_INVENTORY.json'
New-Item -ItemType Directory -Path $PreflightRoot -Force | Out-Null

$env:PYTHONPATH = Join-Path $ProjectRoot 'custom_models\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath $ProjectRoot

& $Python -m benchmark_v2.e9 preflight-all --device cuda --preflight-root $PreflightRoot
$PreflightExitCode = $LASTEXITCODE
if (Test-Path -LiteralPath $InventoryPath -PathType Leaf) {
    $Inventory = Get-Content -LiteralPath $InventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $Inventory.inventory | Select-Object `
        @{Name = 'Model'; Expression = { $_.model_id }}, `
        @{Name = 'Status'; Expression = { $_.status }}, `
        @{Name = 'FailureType'; Expression = {
            $Property = $_.PSObject.Properties['error_type']
            if ($null -ne $Property) { $Property.Value } else { '' }
        }} | Format-Table -AutoSize
    Write-Host "Inventory=$InventoryPath"
} else {
    Write-Error "Preflight inventory was not created: $InventoryPath"
    if ($PreflightExitCode -eq 0) { $PreflightExitCode = 3 }
}
exit $PreflightExitCode
