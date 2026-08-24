# E9 Windows Formal Execution Runbook

## Current recommended workflow

The current E9 formal execution platform is Windows PowerShell with an NVIDIA GPU. The PowerShell files only orchestrate the existing `python -m benchmark_v2.e9` CLI; they do not implement a second training framework.

```text
Windows clone/pull
  -> E9_PRECHECK_WINDOWS.ps1
  -> E9_PREFLIGHT_ALL_6_WINDOWS.ps1
  -> confirm preflight 6/6
  -> E9_RUN_ALL_6_WINDOWS.ps1 (or AUTOSHUTDOWN)
  -> transfer-readiness
  -> workbook-backed readiness 12/12 and pairing 6/6
  -> aggregate --require-complete
  -> formal E9 Excel/Markdown
```

Frozen defaults:

- Project: `D:\PaperProject\GyxPaper2`
- Python: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Analysis: `<ProjectRoot>\custom_models\results\benchmark_v2\msmg_dwu_transfer_seed2026`
- Preflight: `<ProjectRoot>\custom_models\logs\benchmark_v2\e9\preflight`
- Formal logs: `<ProjectRoot>\custom_models\logs\benchmark_v2\e9\formal`

Every path can be overridden. Quoted paths containing spaces are supported.

## 1. Clone or update on the Windows GPU machine

```powershell
git clone https://github.com/efanxu/GyxPaper2.git 'E:\GyxPaper2'
Set-Location 'E:\GyxPaper2'
git switch main
git pull --ff-only origin main

$ProjectRoot = (Get-Location).Path
$Python = 'E:\Miniconda3\envs\env_tslib\python.exe'
$E9 = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\E9'
$Original26Xlsx = 'E:\实验结果\original26_filtered20.xlsx'
```

For the repository's default development paths, set:

```powershell
$ProjectRoot = 'D:\PaperProject\GyxPaper2'
$Python = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$E9 = Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\E9'
$Original26Xlsx = 'C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx'
```

## 2. Precheck (no training)

```powershell
& (Join-Path $E9 'E9_PRECHECK_WINDOWS.ps1') `
    -ProjectRoot $ProjectRoot `
    -Python $Python `
    -Original26Xlsx $Original26Xlsx
```

This prints Git HEAD/branch, Python and Torch versions, CUDA version/availability, GPU name and memory, and validates the exact six-row variant manifest.

## 3. Exact batch4 full-shape preflight

```powershell
& (Join-Path $E9 'E9_PREFLIGHT_ALL_6_WINDOWS.ps1') `
    -ProjectRoot $ProjectRoot `
    -Python $Python

$PreflightInventory = Join-Path $ProjectRoot 'custom_models\logs\benchmark_v2\e9\preflight\E9_PREFLIGHT_INVENTORY.json'
Get-Content -LiteralPath $PreflightInventory -Raw -Encoding UTF8
```

Proceed only when the inventory reports `passed=6`, `expected=6`, and `all_passed=true`. Do not reduce B=4, T=144, N=134, C=16, H=10 or any frozen model setting after a failure.

## 4. Foreground six-model formal run

Without the workbook on the GPU machine, the script still produces transfer readiness:

```powershell
& (Join-Path $E9 'E9_RUN_ALL_6_WINDOWS.ps1') `
    -ProjectRoot $ProjectRoot `
    -Python $Python
```

With the workbook, it additionally produces full readiness:

```powershell
& (Join-Path $E9 'E9_RUN_ALL_6_WINDOWS.ps1') `
    -ProjectRoot $ProjectRoot `
    -Python $Python `
    -Original26Xlsx $Original26Xlsx
```

The model/run list comes only from `E9_VARIANT_MANIFEST.json`. Each transfer uses an independent sequential Python process. Controls, A0/A8, E5, and excluded models are never training targets.

## 5. Background formal run

The repository's native argument joiner preserves spaces, quotes, Unicode, and trailing backslashes:

```powershell
. (Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\WINDOWS_NATIVE_PROCESS_RUNNER.ps1')
$RunScript = Join-Path $E9 'E9_RUN_ALL_6_WINDOWS.ps1'
$Args = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $RunScript,
    '-ProjectRoot', $ProjectRoot, '-Python', $Python,
    '-Original26Xlsx', $Original26Xlsx
)
$Process = Start-Process `
    -FilePath 'powershell.exe' `
    -ArgumentList (Join-GyxWindowsCommandLine -Arguments $Args) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -PassThru
$Process.Id
```

## 6. Background run with automatic shutdown

```powershell
. (Join-Path $ProjectRoot 'custom_models\docs\benchmark_v2\WINDOWS_NATIVE_PROCESS_RUNNER.ps1')
$AutoScript = Join-Path $E9 'E9_RUN_ALL_6_WINDOWS_AUTOSHUTDOWN.ps1'
$Args = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $AutoScript,
    '-ProjectRoot', $ProjectRoot, '-Python', $Python,
    '-Original26Xlsx', $Original26Xlsx
)
$Process = Start-Process `
    -FilePath 'powershell.exe' `
    -ArgumentList (Join-GyxWindowsCommandLine -Arguments $Args) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -PassThru
$Process.Id
```

The wrapper invokes `shutdown.exe /s /t 0` only after run-all exits and the exit code, completed/failed inventories, transfer readiness, optional full readiness, and logs have been saved.

## 7. Monitoring

```powershell
$FormalLogRoot = Join-Path $ProjectRoot 'custom_models\logs\benchmark_v2\e9\formal'
Get-Content -LiteralPath (Join-Path $FormalLogRoot 'run_all_6.log') -Tail 50 -Wait
Get-Content -LiteralPath (Join-Path $FormalLogRoot 'LightTS_node_shared_chunk8_bs4_msmg_dwu_seed2026.log') -Tail 50 -Wait
Get-Content -LiteralPath (Join-Path $FormalLogRoot 'run_all.pid')
Get-Process powershell, python -ErrorAction SilentlyContinue
nvidia-smi
```

Status inventories:

```powershell
Get-Content -LiteralPath (Join-Path $FormalLogRoot 'completed_models.csv')
Get-Content -LiteralPath (Join-Path $FormalLogRoot 'failed_models.csv')
Get-Content -LiteralPath (Join-Path $FormalLogRoot 'run_all_exit_code.txt')
Get-Content -LiteralPath (Join-Path $ProjectRoot 'custom_models\results\benchmark_v2\msmg_dwu_transfer_seed2026\E9_TRANSFER_READINESS.json') -Raw -Encoding UTF8
```

## 8. Safe continuation after interruption

Rerun the same foreground or background command. A COMPLETED identity-matching transfer is safely skipped. A matching RUNNING/FAILED run with `last_checkpoint.pt` resumes model, optimizer, AMP scaler, MS-MG-DWU state, RNG, epoch, early-stopping state, and history. A matching TRAINING_COMPLETED/EARLY_STOPPED run resumes evaluation. Identity conflicts or incomplete directories without a safe checkpoint fail closed and are never deleted or overwritten.

## 9. Readiness and aggregate

Artifact-tool is needed only for the final Excel export. Supply its runtime through parameters or environment variables:

```powershell
$ArtifactNode = 'C:\path\to\node.exe'
$ArtifactNodeModules = 'C:\path\to\node_modules'

& (Join-Path $E9 'E9_READINESS_AND_AGGREGATE_WINDOWS.ps1') `
    -ProjectRoot $ProjectRoot `
    -Python $Python `
    -Original26Xlsx $Original26Xlsx `
    -ArtifactNode $ArtifactNode `
    -ArtifactNodeModules $ArtifactNodeModules
```

The script stops nonzero and prints missing pairs unless `CORE_E9_READY=12/12` and `LOSS_ONLY_PAIRING_READY=6/6`. Only then does it execute `aggregate --require-complete`.

## 10. Linux compatibility path

The existing Linux scripts remain supported but are no longer the current recommended workflow:

```bash
cd /root/autodl-tmp/GyxPaper2
bash custom_models/docs/benchmark_v2/E9/E9_PRECHECK_LINUX.sh
bash custom_models/docs/benchmark_v2/E9/E9_PREFLIGHT_ALL_6_LINUX.sh
bash custom_models/docs/benchmark_v2/E9/E9_RUN_ALL_6_LINUX.sh
```

The Linux autoshutdown and readiness/aggregate wrappers remain in the same E9 directory.
