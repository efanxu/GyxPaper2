# E2-A formal runbook

This file prepares commands only. No E2-A formal run or RTX 4060 preflight was executed.

Transformer and PatchTST automatically preflight before formal training. iTransformer and TimeXer passed exact local full-shape and launch a new formal worker directly. Every `train` request uses one model per Python process.

## Windows PyCharm

- Interpreter: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script path: `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment: `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

Create one PyCharm configuration per Script-parameters line:

```text
train --model transformer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026 --run-id Transformer_node_shared_d512_seed2026 --device cuda
train --model patchtst --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026 --run-id PatchTST_node_shared_p16_s8_seed2026 --device cuda
train --model itransformer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026 --run-id iTransformer_node_shared_d512_seed2026 --device cuda
train --model timexer --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026 --run-id TimeXer_node_shared_ms_d512_seed2026 --device cuda
```

Standalone preflight Script parameters:

```text
hardware-preflight --model transformer
hardware-preflight --model patchtst
hardware-preflight --model itransformer
hardware-preflight --model timexer
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
```

Standalone exact preflight:

```powershell
& $python $runner hardware-preflight --model transformer
& $python $runner hardware-preflight --model patchtst
& $python $runner hardware-preflight --model itransformer
& $python $runner hardware-preflight --model timexer
```

One-command formal requests:

```powershell
& $python $runner train --model transformer --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026' --run-id Transformer_node_shared_d512_seed2026 --device cuda
& $python $runner train --model patchtst --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026' --run-id PatchTST_node_shared_p16_s8_seed2026 --device cuda
& $python $runner train --model itransformer --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026' --run-id iTransformer_node_shared_d512_seed2026 --device cuda
& $python $runner train --model timexer --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e2_a_seed2026' --run-id TimeXer_node_shared_ms_d512_seed2026 --device cuda
```

Check processes/GPU and last exit code:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_benchmark.py' } | Select-Object ProcessId,ParentProcessId,CommandLine
nvidia-smi
$LASTEXITCODE
```

Preflight artifacts:

- `custom_models/results_smoke/benchmark_v2/hardware_preflight/transformer/76ea8eedcd747b1e3c67/hardware_preflight.json`
- `custom_models/results_smoke/benchmark_v2/hardware_preflight/patchtst/88c28fc94eac08f63f42/hardware_preflight.json`
- optional local rechecks follow the corresponding record-scoped iTransformer/TimeXer directory.

Formal outputs are below `custom_models/results/benchmark_v2/e2_a_seed2026/<run-id>`.

## Linux foreground

From the project checkout:

```bash
export PYTHONPATH="$(pwd)/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/benchmark_v2

python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model transformer
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model patchtst
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model itransformer
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model timexer

python custom_models/src/benchmark_v2/run_benchmark.py train --model transformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id Transformer_node_shared_d512_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model patchtst --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id PatchTST_node_shared_p16_s8_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model itransformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id iTransformer_node_shared_d512_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model timexer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id TimeXer_node_shared_ms_d512_seed2026 --device cuda
```

## Linux nohup, logs, exit code, sync, unconditional shutdown

Run only one of these at a time:

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model transformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id Transformer_node_shared_d512_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/Transformer_node_shared_d512_seed2026.log 2>&1 &

nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model patchtst --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id PatchTST_node_shared_p16_s8_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/PatchTST_node_shared_p16_s8_seed2026.log 2>&1 &

nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model itransformer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id iTransformer_node_shared_d512_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/iTransformer_node_shared_d512_seed2026.log 2>&1 &

nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model timexer --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e2_a_seed2026 --run-id TimeXer_node_shared_ms_d512_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/TimeXer_node_shared_ms_d512_seed2026.log 2>&1 &
```

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
nvidia-smi
wait <pid>; echo $?
sync
```

Safety status: RTX 4060 preflight `NOT_RUN`; E2-A formal Full `NOT_RUN`; E2-B `NOT_STARTED`.
