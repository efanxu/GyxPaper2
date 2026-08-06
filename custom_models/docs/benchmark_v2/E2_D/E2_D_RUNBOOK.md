# E2-D runbook

This document only supplies commands. None of the formal commands or
target-machine preflights below were executed during E2-D.

Formal output root:

`D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_d_seed2026`

Frozen run IDs:

- `Crossformer_node_shared_seg12_win2_d32_el2_seed2026`
- `MSGNet_node_shared_k5_d512_el2_gdep2_seed2026`
- `TimeFilter_node_shared_p16_d512_el2_tp0p5_seed2026`

Run one model in one Python process at a time.

## Windows PyCharm

- Interpreter:
  `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script:
  `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment:
  `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

Create three independent preflight configurations:

```text
hardware-preflight --model crossformer
hardware-preflight --model msgnet
hardware-preflight --model timefilter
```

Create three independent formal train configurations:

```text
train --model crossformer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_d_seed2026 --run-id Crossformer_node_shared_seg12_win2_d32_el2_seed2026 --device cuda
train --model msgnet --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_d_seed2026 --run-id MSGNet_node_shared_k5_d512_el2_gdep2_seed2026 --device cuda
train --model timefilter --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_d_seed2026 --run-id TimeFilter_node_shared_p16_d512_el2_tp0p5_seed2026 --device cuda
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
$output='D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_d_seed2026'
```

Standalone exact preflight:

```powershell
& $python $runner hardware-preflight --model crossformer
& $python $runner hardware-preflight --model msgnet
& $python $runner hardware-preflight --model timefilter
```

Formal requests:

```powershell
& $python $runner train --model crossformer --input-path $input --target-path $target --output-root $output --run-id 'Crossformer_node_shared_seg12_win2_d32_el2_seed2026' --device cuda
& $python $runner train --model msgnet --input-path $input --target-path $target --output-root $output --run-id 'MSGNet_node_shared_k5_d512_el2_gdep2_seed2026' --device cuda
& $python $runner train --model timefilter --input-path $input --target-path $target --output-root $output --run-id 'TimeFilter_node_shared_p16_d512_el2_tp0p5_seed2026' --device cuda
```

MSGNet and TimeFilter automatically start an independent exact preflight when
no matching PASS exists. The preflight worker must exit before a new formal
worker starts. `FAIL_OOM` or `FAIL_NON_OOM` creates no formal run. Crossformer
does not require a formal preflight, but a standalone command is supplied for
uniform operational verification.

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

python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model crossformer
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model msgnet
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model timefilter

python custom_models/src/benchmark_v2/run_benchmark.py train --model crossformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_d_seed2026 --run-id Crossformer_node_shared_seg12_win2_d32_el2_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model msgnet --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_d_seed2026 --run-id MSGNet_node_shared_k5_d512_el2_gdep2_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model timefilter --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_d_seed2026 --run-id TimeFilter_node_shared_p16_d512_el2_tp0p5_seed2026 --device cuda
```

## Linux nohup, exit code, sync, and unconditional shutdown

Run exactly one block at a time.

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model crossformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_d_seed2026 --run-id Crossformer_node_shared_seg12_win2_d32_el2_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/Crossformer_node_shared_seg12_win2_d32_el2_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model msgnet --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_d_seed2026 --run-id MSGNet_node_shared_k5_d512_el2_gdep2_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/MSGNet_node_shared_k5_d512_el2_gdep2_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model timefilter --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_d_seed2026 --run-id TimeFilter_node_shared_p16_d512_el2_tp0p5_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/TimeFilter_node_shared_p16_d512_el2_tp0p5_seed2026.log 2>&1 &
```

Inspect:

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
nvidia-smi
```

A reusable preflight PASS must exactly match model ID, model-config record,
Protocol record, source record, B/T/N/C/H, AMP, completed forward/backward, and
`status=PASS`.
