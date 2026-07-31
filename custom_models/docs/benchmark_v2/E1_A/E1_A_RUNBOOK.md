# E1-A formal runbook

These commands are prepared but were not executed in E1-A.

Formal output root:

`custom_models/results/benchmark_v2/basic_lightweight_seed2026`

Run IDs:

- `Persistence_seed2026`
- `MovingAverage_w144_seed2026`
- `GRU_node_shared_h64_l1_seed2026`

## Windows PyCharm

Common settings for all three configurations:

- Interpreter: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script path: `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment variables: `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

Persistence script parameters:

```text
evaluate-only --model persistence --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id Persistence_seed2026 --device cpu
```

MovingAverage script parameters:

```text
evaluate-only --model moving_average --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id MovingAverage_w144_seed2026 --device cpu
```

GRU script parameters:

```text
train --model gru --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id GRU_node_shared_h64_l1_seed2026 --device cuda
```

Each configuration writes only its named run directory below the formal output root.

## Windows PowerShell

Set the environment once:

```powershell
$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES='0'
```

Persistence:

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' evaluate-only --model persistence --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026' --run-id Persistence_seed2026 --device cpu
```

MovingAverage:

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' evaluate-only --model moving_average --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026' --run-id MovingAverage_w144_seed2026 --device cpu
```

GRU:

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' train --model gru --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026' --run-id GRU_node_shared_h64_l1_seed2026 --device cuda
```

## Linux

Do not guess the project path. First enter the project checkout and verify:

```bash
pwd
test -f custom_models/src/benchmark_v2/run_benchmark.py
export PYTHONPATH="$(pwd)/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/benchmark_v2
```

### Persistence

Foreground:

```bash
python custom_models/src/benchmark_v2/run_benchmark.py evaluate-only --model persistence --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id Persistence_seed2026 --device cpu
```

Foreground with unconditional shutdown after Python exits:

```bash
python custom_models/src/benchmark_v2/run_benchmark.py evaluate-only --model persistence --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id Persistence_seed2026 --device cpu; /usr/bin/shutdown -h now
```

Background with log, preserved exit code, sync, and unconditional shutdown:

```bash
nohup bash -lc 'python custom_models/src/benchmark_v2/run_benchmark.py evaluate-only --model persistence --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id Persistence_seed2026 --device cpu; code=$?; sync; /usr/bin/shutdown -h now; exit $code' > logs/benchmark_v2/Persistence_seed2026.log 2>&1 &
```

### MovingAverage

Foreground:

```bash
python custom_models/src/benchmark_v2/run_benchmark.py evaluate-only --model moving_average --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id MovingAverage_w144_seed2026 --device cpu
```

Foreground with unconditional shutdown:

```bash
python custom_models/src/benchmark_v2/run_benchmark.py evaluate-only --model moving_average --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id MovingAverage_w144_seed2026 --device cpu; /usr/bin/shutdown -h now
```

Background:

```bash
nohup bash -lc 'python custom_models/src/benchmark_v2/run_benchmark.py evaluate-only --model moving_average --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id MovingAverage_w144_seed2026 --device cpu; code=$?; sync; /usr/bin/shutdown -h now; exit $code' > logs/benchmark_v2/MovingAverage_w144_seed2026.log 2>&1 &
```

### GRU

Foreground:

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model gru --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id GRU_node_shared_h64_l1_seed2026 --device cuda
```

Foreground with unconditional shutdown:

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model gru --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id GRU_node_shared_h64_l1_seed2026 --device cuda; /usr/bin/shutdown -h now
```

Background:

```bash
nohup bash -lc 'python custom_models/src/benchmark_v2/run_benchmark.py train --model gru --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id GRU_node_shared_h64_l1_seed2026 --device cuda; code=$?; sync; /usr/bin/shutdown -h now; exit $code' > logs/benchmark_v2/GRU_node_shared_h64_l1_seed2026.log 2>&1 &
```

Log and process inspection:

```bash
tail -f logs/benchmark_v2/GRU_node_shared_h64_l1_seed2026.log
ps -ef | grep '[r]un_benchmark.py'
```
