# E2-B formal runbook

Commands are prepared only. No formal E2-B run or target high-memory preflight was executed.

Final Run IDs:

- `TimesNet_node_shared_d32_k5_seed2026`
- `MICN_node_shared_k12_16_d32_seed2026`
- `WPMixer_node_shared_db2_l1_p16_s8_d256_seed2026`
- `MultiPatchFormer_node_shared_p8_16_24_32_d256_seed2026`

Formal output root: `custom_models/results/benchmark_v2/e2_b_seed2026` (not created by E2-B).

TimesNet and MICN passed exact local full-shape. WPMixer and MultiPatchFormer automatically run exact preflight before training. Each train request launches a new formal worker; preflight workers exit first and release their CUDA contexts.

## Windows PyCharm

- Interpreter: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script path: `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment: `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

Create one configuration for each full Script parameters line:

```text
train --model timesnet --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_b_seed2026 --run-id TimesNet_node_shared_d32_k5_seed2026 --device cuda
train --model micn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_b_seed2026 --run-id MICN_node_shared_k12_16_d32_seed2026 --device cuda
train --model wpmixer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_b_seed2026 --run-id WPMixer_node_shared_db2_l1_p16_s8_d256_seed2026 --device cuda
train --model multipatchformer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_b_seed2026 --run-id MultiPatchFormer_node_shared_p8_16_24_32_d256_seed2026 --device cuda
```

Standalone preflight parameters:

```text
hardware-preflight --model timesnet
hardware-preflight --model micn
hardware-preflight --model wpmixer
hardware-preflight --model multipatchformer
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
$output='D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_b_seed2026'
```

Standalone exact preflight:

```powershell
& $python $runner hardware-preflight --model timesnet
& $python $runner hardware-preflight --model micn
& $python $runner hardware-preflight --model wpmixer
& $python $runner hardware-preflight --model multipatchformer
```

One-command formal requests:

```powershell
& $python $runner train --model timesnet --input-path $input --target-path $target --output-root $output --run-id 'TimesNet_node_shared_d32_k5_seed2026' --device cuda
& $python $runner train --model micn --input-path $input --target-path $target --output-root $output --run-id 'MICN_node_shared_k12_16_d32_seed2026' --device cuda
& $python $runner train --model wpmixer --input-path $input --target-path $target --output-root $output --run-id 'WPMixer_node_shared_db2_l1_p16_s8_d256_seed2026' --device cuda
& $python $runner train --model multipatchformer --input-path $input --target-path $target --output-root $output --run-id 'MultiPatchFormer_node_shared_p8_16_24_32_d256_seed2026' --device cuda
```

WPMixer/MultiPatchFormer train commands automatically preflight, stop with nonzero exit on `FAIL_OOM`/`FAIL_NON_OOM`, and create no formal run before PASS. Do not edit Registry manually.

Inspect:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_benchmark.py' } | Select-Object ProcessId,ParentProcessId,CommandLine
nvidia-smi
$LASTEXITCODE
```

## Linux preparation and foreground

```bash
pwd
test -f custom_models/src/benchmark_v2/run_benchmark.py
export PYTHONPATH="$(pwd)/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/benchmark_v2

python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model timesnet
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model micn
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model wpmixer
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model multipatchformer

python custom_models/src/benchmark_v2/run_benchmark.py train --model timesnet --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id TimesNet_node_shared_d32_k5_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model micn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id MICN_node_shared_k12_16_d32_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model wpmixer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id WPMixer_node_shared_db2_l1_p16_s8_d256_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model multipatchformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id MultiPatchFormer_node_shared_p8_16_24_32_d256_seed2026 --device cuda
```

Run only one model at a time.

## Linux nohup with unconditional shutdown

Use exactly one block at a time:

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model timesnet --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id TimesNet_node_shared_d32_k5_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/TimesNet_node_shared_d32_k5_seed2026.log 2>&1 &

nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model micn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id MICN_node_shared_k12_16_d32_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/MICN_node_shared_k12_16_d32_seed2026.log 2>&1 &

nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model wpmixer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id WPMixer_node_shared_db2_l1_p16_s8_d256_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/WPMixer_node_shared_db2_l1_p16_s8_d256_seed2026.log 2>&1 &

nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model multipatchformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_b_seed2026 --run-id MultiPatchFormer_node_shared_p8_16_24_32_d256_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/MultiPatchFormer_node_shared_p8_16_24_32_d256_seed2026.log 2>&1 &
```

Foreground without shutdown is the earlier `python ... train` command. Nohup without shutdown is the same block with `sync`, `/usr/bin/shutdown`, and their surrounding lines removed.

Inspect logs/process/GPU:

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
nvidia-smi
```

Expected preflight artifacts:

`custom_models/results_smoke/benchmark_v2/hardware_preflight/<model>/<identity-record>/hardware_preflight.json`.

A PASS must exactly match model/config/protocol/source records, B/T/N/C/H, AMP, completed forward, completed backward, and status.
