[CmdletBinding()]
param(
    [ValidateSet('All', 'Preflight', 'Formal')]
    [string]$Mode = 'All',

    [string]$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe',

    [int[]]$Seeds = @(2026),

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [string]$OutputRoot = 'custom_models/results/empirical_analysis_v1',

    [string]$Profile = 'STMG_FORMAL_V2',

    [string]$StateDirectory = 'custom_models/logs/steps_3_to_8_pipeline',

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$SourceRoot = Join-Path $ProjectRoot 'custom_models\src'
$PythonCommand = Get-Command $PythonExecutable -ErrorAction Stop
$Python = $PythonCommand.Source

if (-not [IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot $OutputRoot
}
if (-not [IO.Path]::IsPathRooted($StateDirectory)) {
    $StateDirectory = Join-Path $ProjectRoot $StateDirectory
}

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
if (-not $PlanOnly) {
    New-Item -ItemType Directory -Force -Path $StateDirectory | Out-Null
}

function Invoke-CheckpointedStage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $markerPath = Join-Path $StateDirectory ($Name + '.done')
    $displayCommand = '& ' + ('"{0}"' -f $Python) + ' ' + (($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' ')

    if ($PlanOnly) {
        Write-Host "[PLAN] $Name"
        Write-Host $displayCommand
        return
    }
    if (Test-Path -LiteralPath $markerPath) {
        Write-Host "[SKIP] $Name (stage marker exists: $markerPath)"
        return
    }

    Write-Host "[RUN] $Name"
    Write-Host $displayCommand
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Name' failed with exit code $LASTEXITCODE. Re-run the same command to resume."
    }

    @(
        "completed_at=$([DateTimeOffset]::Now.ToString('o'))"
        "python=$Python"
        "command=$displayCommand"
    ) | Set-Content -LiteralPath $markerPath -Encoding utf8
    Write-Host "[DONE] $Name"
}

$Families = @(
    [pscustomobject]@{
        Step = 3
        Family = 'T'
        Scope = 'internal_mechanism'
        AllVariants = @('T0', 'T1', 'T2', 'T3', 'T4', 'T5')
        FormalVariants = @('T1', 'T2', 'T3', 'T4', 'T5')
    },
    [pscustomobject]@{
        Step = 4
        Family = 'G'
        Scope = 'internal_graph'
        AllVariants = @('G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8')
        FormalVariants = @('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8')
    },
    [pscustomobject]@{
        Step = 5
        Family = 'D'
        Scope = 'internal_diffusion'
        AllVariants = @('D0', 'D1', 'D2', 'D3', 'D4', 'D5')
        FormalVariants = @('D0', 'D1', 'D2', 'D3', 'D5')
    },
    [pscustomobject]@{
        Step = 6
        Family = 'F'
        Scope = 'internal_fusion'
        AllVariants = @('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8')
        FormalVariants = @('F0', 'F1', 'F2', 'F3', 'F4', 'F5')
    },
    [pscustomobject]@{
        Step = 7
        Family = 'N'
        Scope = 'internal_decoder'
        AllVariants = @('N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8')
        FormalVariants = @('N0', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7')
    },
    [pscustomobject]@{
        Step = 8
        Family = 'L'
        Scope = 'internal_loss'
        AllVariants = @('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7')
        FormalVariants = @('L0', 'L1', 'L2', 'L4', 'L5', 'L6')
    }
)

if ($Mode -in @('All', 'Preflight')) {
    Invoke-CheckpointedStage -Name 'preflight_compile' -Arguments @(
        '-m', 'compileall', 'custom_models/src/st_mgprompt'
    )

    Invoke-CheckpointedStage -Name 'preflight_tests_steps_3_to_8' -Arguments @(
        '-m', 'pytest',
        'custom_models/tests/empirical_analysis/test_step3_fixed_dual.py',
        'custom_models/tests/empirical_analysis/test_step4_graph_semantics.py',
        'custom_models/tests/empirical_analysis/test_step5_diffusion.py',
        'custom_models/tests/empirical_analysis/test_step6_fusion.py',
        'custom_models/tests/empirical_analysis/test_step7_prompt_decoder.py',
        'custom_models/tests/empirical_analysis/test_step8_losses.py',
        '-q'
    )

    foreach ($item in $Families) {
        $baseArguments = @(
            '-m', 'st_mgprompt.run_empirical',
            '--family', $item.Family,
            '--variants'
        ) + $item.AllVariants + @(
            '--profile', $Profile,
            '--source-scope', $item.Scope,
            '--output-root', $OutputRoot,
            '--device', $Device
        )

        Invoke-CheckpointedStage -Name ("step{0}_dry_run" -f $item.Step) -Arguments ($baseArguments + @('--dry-run'))
        Invoke-CheckpointedStage -Name ("step{0}_smoke" -f $item.Step) -Arguments (
            $baseArguments + @('--smoke', '--resume', '--skip-completed')
        )
        Invoke-CheckpointedStage -Name ("step{0}_full_shape" -f $item.Step) -Arguments ($baseArguments + @('--full-shape'))
    }
}

if ($Mode -in @('All', 'Formal')) {
    $SeedArguments = @($Seeds | ForEach-Object { $_.ToString() })
    foreach ($item in $Families) {
        $formalArguments = @(
            '-m', 'st_mgprompt.run_empirical',
            '--family', $item.Family,
            '--variants'
        ) + $item.FormalVariants + @(
            '--seeds'
        ) + $SeedArguments + @(
            '--profile', $Profile,
            '--source-scope', $item.Scope,
            '--output-root', $OutputRoot,
            '--device', $Device,
            '--run-full',
            '--resume',
            '--skip-completed'
        )

        Invoke-CheckpointedStage -Name ("step{0}_formal" -f $item.Step) -Arguments $formalArguments
    }
}

if (-not $PlanOnly) {
    Write-Host 'Steps 3-8 pipeline completed successfully.'
    Write-Host "Results: $OutputRoot"
    Write-Host "Stage markers: $StateDirectory"
}
