# E9 Linux/Windows Runbook

## 1. Enter the Linux project

```bash
cd /root/autodl-tmp/GyxPaper2
export PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH=/root/autodl-tmp/GyxPaper2/custom_models/src
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
```

If the workbook is present on Linux, set it explicitly:

```bash
export ORIGINAL26_XLSX=/root/autodl-tmp/GyxPaper2/original26_filtered20.xlsx
```

## 2. Precheck

```bash
bash custom_models/docs/benchmark_v2/E9/E9_PRECHECK_LINUX.sh
```

## 3. Full-shape batch4 preflight inventory

This checks all six models and records every failure instead of stopping at the first.

```bash
bash custom_models/docs/benchmark_v2/E9/E9_PREFLIGHT_ALL_6_LINUX.sh
```

Inspect:

```bash
cat custom_models/logs/benchmark_v2/e9/preflight/E9_PREFLIGHT_INVENTORY.json
```

Do not reduce batch, node count, lookback, horizon, hidden size, layers, patches, or graph settings after OOM.

## 4. Foreground six-model formal run

```bash
bash custom_models/docs/benchmark_v2/E9/E9_RUN_ALL_6_LINUX.sh
```

The executed list comes only from `E9_VARIANT_MANIFEST.json`. It trains one transfer per process and never trains controls or internal references.

At the end it always writes `E9_TRANSFER_READINESS.json`. This is a strict six-transfer artifact/config status, not a substitute for the workbook-backed `E9_READINESS.json`. Exporting `ORIGINAL26_XLSX` also runs the latter.

## 5. Background nohup

```bash
mkdir -p custom_models/logs/benchmark_v2/e9/formal
nohup bash custom_models/docs/benchmark_v2/E9/E9_RUN_ALL_6_LINUX.sh \
  > custom_models/logs/benchmark_v2/e9/formal/nohup_run_all.log 2>&1 &
echo $! > custom_models/logs/benchmark_v2/e9/formal/nohup_run_all.pid
```

## 6. Background nohup with automatic shutdown

```bash
mkdir -p custom_models/logs/benchmark_v2/e9/formal
nohup bash custom_models/docs/benchmark_v2/E9/E9_RUN_ALL_6_LINUX_AUTOSHUTDOWN.sh \
  > custom_models/logs/benchmark_v2/e9/formal/nohup_autoshutdown.log 2>&1 &
echo $! > custom_models/logs/benchmark_v2/e9/formal/nohup_autoshutdown.pid
```

Shutdown occurs only after the complete run-all flow exits and saves the overall code, completed list, failure list, transfer readiness, optional full readiness, and logs.

## 7. Monitor

```bash
tail -f custom_models/logs/benchmark_v2/e9/formal/run_all_6.log
tail -f custom_models/logs/benchmark_v2/e9/formal/LightTS_node_shared_chunk8_bs4_msmg_dwu_seed2026.log
cat custom_models/logs/benchmark_v2/e9/formal/nohup_run_all.pid
ps -ef | grep '[b]enchmark_v2.e9'
ps -ef | grep '[r]un_benchmark'
watch -n 2 nvidia-smi
cat custom_models/logs/benchmark_v2/e9/formal/run_all_exit_code.txt
cat custom_models/logs/benchmark_v2/e9/formal/failed_models.csv
cat custom_models/logs/benchmark_v2/e9/formal/completed_models.csv
```

Per-model logs are under `custom_models/logs/benchmark_v2/e9/formal/` and use the transfer run-id as filename.

## 8. Linux readiness/aggregate when artifact-tool runtime is available

```bash
bash custom_models/docs/benchmark_v2/E9/E9_READINESS_AND_AGGREGATE_LINUX.sh "$ORIGINAL26_XLSX"
```

Formal Excel generation additionally requires `E9_ARTIFACT_NODE` and `E9_ARTIFACT_NODE_MODULES` to point to the approved artifact-tool runtime. If that runtime is not installed on the cloud host, run readiness there, sync results to Windows, and create the final report on Windows.

## 9. After syncing results back to Windows

```powershell
Set-Location 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:E9_ARTIFACT_NODE='C:\Users\12811\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$env:E9_ARTIFACT_NODE_MODULES='C:\Users\12811\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m benchmark_v2.e9 readiness --original26-xlsx 'C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx'
Get-Content -Raw -Encoding UTF8 'custom_models\results\benchmark_v2\msmg_dwu_transfer_seed2026\E9_READINESS.json'
```

Only after `CORE_E9_READY=12/12` and `LOSS_ONLY_PAIRING_READY=6/6`:

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m benchmark_v2.e9 aggregate --original26-xlsx 'C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx' --require-complete
```

The strict command exits nonzero and does not create the formal workbook while evidence is incomplete.
