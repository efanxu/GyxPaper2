# E3-B formal runbook

These commands are prepared only. No formal Full, formal evaluation, or target
high-memory machine preflight was executed in E3-B.

Formal output root (not created):

`D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_b_seed2026`

Frozen run IDs:

- `GCN_native_agcn_h64_l2_seed2026`
- `STGCN_native_cheb3_kt3_b2_seed2026`
- `DCRNN_native_dualrw_k2_h64_l2_ar_seed2026`

Run one model in one Python process at a time.

## Windows PyCharm

- Interpreter:
  `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script:
  `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment:
  `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

Create three independent diagnostic preflight configurations:

```text
hardware-preflight --model gcn
hardware-preflight --model stgcn
hardware-preflight --model dcrnn
```

All three passed local exact full-shape and do not require a mandatory formal
preflight. The standalone commands remain useful for deployment diagnostics.

Create three independent train configurations:

```text
train --model gcn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_b_seed2026 --run-id GCN_native_agcn_h64_l2_seed2026 --device cuda
train --model stgcn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_b_seed2026 --run-id STGCN_native_cheb3_kt3_b2_seed2026 --device cuda
train --model dcrnn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_b_seed2026 --run-id DCRNN_native_dualrw_k2_h64_l2_ar_seed2026 --device cuda
```

## Windows PowerShell

```powershell
$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES='0'
$python='D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$runner='D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py'
$input='D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet'
$target='D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet'
$output='D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_b_seed2026'
```

Standalone exact preflight:

```powershell
& $python $runner hardware-preflight --model gcn
& $python $runner hardware-preflight --model stgcn
& $python $runner hardware-preflight --model dcrnn
```

Independent formal requests:

```powershell
& $python $runner train --model gcn --input-path $input --target-path $target --output-root $output --run-id 'GCN_native_agcn_h64_l2_seed2026' --device cuda
& $python $runner train --model stgcn --input-path $input --target-path $target --output-root $output --run-id 'STGCN_native_cheb3_kt3_b2_seed2026' --device cuda
& $python $runner train --model dcrnn --input-path $input --target-path $target --output-root $output --run-id 'DCRNN_native_dualrw_k2_h64_l2_ar_seed2026' --device cuda
```

Inspect:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_benchmark.py' } | Select-Object ProcessId,ParentProcessId,CommandLine
nvidia-smi
$LASTEXITCODE
```

## Linux foreground

```bash
cd /path/to/GyxPaper2
test -f custom_models/src/benchmark_v2/run_benchmark.py
export PYTHONPATH="$(pwd)/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/benchmark_v2

python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model gcn
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model stgcn
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model dcrnn

python custom_models/src/benchmark_v2/run_benchmark.py train --model gcn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_b_seed2026 --run-id GCN_native_agcn_h64_l2_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model stgcn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_b_seed2026 --run-id STGCN_native_cheb3_kt3_b2_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model dcrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_b_seed2026 --run-id DCRNN_native_dualrw_k2_h64_l2_ar_seed2026 --device cuda
```

## Linux nohup, exit code, sync, and unconditional shutdown

Run exactly one block at a time.

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model gcn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_b_seed2026 --run-id GCN_native_agcn_h64_l2_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/GCN_native_agcn_h64_l2_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model stgcn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_b_seed2026 --run-id STGCN_native_cheb3_kt3_b2_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/STGCN_native_cheb3_kt3_b2_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model dcrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_b_seed2026 --run-id DCRNN_native_dualrw_k2_h64_l2_ar_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/DCRNN_native_dualrw_k2_h64_l2_ar_seed2026.log 2>&1 &
```

Inspect:

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
nvidia-smi
```

A reusable graph preflight PASS must match model/config/benchmark/source records,
exact B/T/N/C/H, AMP, graph protocol, graph bundle, node order, location source,
support names/records, and completed forward/backward with `status=PASS`.
