# E2-C formal runbook

Commands are prepared only. No command in this runbook was executed as a
formal run or target-machine preflight during E2-C.

Formal output root (not created):

`D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_c_seed2026`

Stable run IDs:

- `TimeMixer_node_shared_ci1_ds3_w2_avg_d16_seed2026`
- `TSMixer_node_shared_el2_d32_seed2026`
- `FreTS_node_shared_tcfft_e128_h256_seed2026`

`tcfft` is accurate: final FreTS string `"0"` activates both temporal and
channel frequency learners. TimeMixer and FreTS require an exact matching PASS
on the target machine. TSMixer already passed locally, but a standalone
preflight command is supplied for uniform operational checking.

## Windows PyCharm

- Interpreter:
  `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script:
  `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment:
  `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

Create separate configurations with one parameters line each:

```text
hardware-preflight --model timemixer
hardware-preflight --model tsmixer
hardware-preflight --model frets
```

```text
train --model timemixer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_c_seed2026 --run-id TimeMixer_node_shared_ci1_ds3_w2_avg_d16_seed2026 --device cuda
train --model tsmixer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_c_seed2026 --run-id TSMixer_node_shared_el2_d32_seed2026 --device cuda
train --model frets --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_c_seed2026 --run-id FreTS_node_shared_tcfft_e128_h256_seed2026 --device cuda
```

Run only one Python process at a time.

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
$output='D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_c_seed2026'
```

Standalone exact preflight:

```powershell
& $python $runner hardware-preflight --model timemixer
& $python $runner hardware-preflight --model tsmixer
& $python $runner hardware-preflight --model frets
```

Independent formal requests:

```powershell
& $python $runner train --model timemixer --input-path $input --target-path $target --output-root $output --run-id 'TimeMixer_node_shared_ci1_ds3_w2_avg_d16_seed2026' --device cuda
& $python $runner train --model tsmixer --input-path $input --target-path $target --output-root $output --run-id 'TSMixer_node_shared_el2_d32_seed2026' --device cuda
& $python $runner train --model frets --input-path $input --target-path $target --output-root $output --run-id 'FreTS_node_shared_tcfft_e128_h256_seed2026' --device cuda
```

For TimeMixer/FreTS, `train` launches preflight when no exact PASS exists,
waits for that process to exit, and only then launches a new formal worker.
`FAIL_OOM` or `FAIL_NON_OOM` returns nonzero and creates no formal run.

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

python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model timemixer
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model tsmixer
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model frets

python custom_models/src/benchmark_v2/run_benchmark.py train --model timemixer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_c_seed2026 --run-id TimeMixer_node_shared_ci1_ds3_w2_avg_d16_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model tsmixer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_c_seed2026 --run-id TSMixer_node_shared_el2_d32_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model frets --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_c_seed2026 --run-id FreTS_node_shared_tcfft_e128_h256_seed2026 --device cuda
```

Run one foreground command at a time.

## Linux nohup with exit code, sync, and unconditional shutdown

Use exactly one block at a time:

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model timemixer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_c_seed2026 --run-id TimeMixer_node_shared_ci1_ds3_w2_avg_d16_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/TimeMixer_node_shared_ci1_ds3_w2_avg_d16_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model tsmixer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_c_seed2026 --run-id TSMixer_node_shared_el2_d32_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/TSMixer_node_shared_el2_d32_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model frets --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_c_seed2026 --run-id FreTS_node_shared_tcfft_e128_h256_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/FreTS_node_shared_tcfft_e128_h256_seed2026.log 2>&1 &
```

Inspect:

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
nvidia-smi
```

Expected PASS artifact root:

`custom_models/results_smoke/benchmark_v2/hardware_preflight/<model>/<identity-record>/hardware_preflight.json`

A reusable PASS must match model ID, model-config record, protocol record, source
record, `B/T/N/C/H`, AMP, forward/backward completion, and `status=PASS`.
