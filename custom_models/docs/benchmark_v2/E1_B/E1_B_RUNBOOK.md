# E1-B formal runbook

本文件只准备命令；E1-B 未执行任何正式训练。当前 DLinear/LightTS 为 formal-ready；TiDE/SegRNN 因 exact full-shape OOM 被 Registry fail-closed，以下两条命令仅保留复现配置，解除显存阻塞前会返回错误且不会创建正式 run。

正式输出根：

`custom_models/results/benchmark_v2/basic_lightweight_seed2026`

Run IDs：

- `DLinear_node_shared_ma25_seed2026`
- `LightTS_node_shared_chunk8_seed2026`
- `TiDE_node_shared_d512_seed2026`
- `SegRNN_node_shared_seg2_d512_seed2026`

## Windows PyCharm

四个配置的共同设置：

- Interpreter: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script path: `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment variables: `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

DLinear parameters:

```text
train --model dlinear --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id DLinear_node_shared_ma25_seed2026 --device cuda
```

LightTS parameters:

```text
train --model lightts --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id LightTS_node_shared_chunk8_seed2026 --device cuda
```

TiDE parameters（当前 blocked）：

```text
train --model tide --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda
```

SegRNN parameters（当前 blocked）：

```text
train --model segrnn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda
```

## Windows PowerShell

先设置环境：

```powershell
$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES='0'
```

四条完整命令：

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' train --model dlinear --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026' --run-id DLinear_node_shared_ma25_seed2026 --device cuda

& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' train --model lightts --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026' --run-id LightTS_node_shared_chunk8_seed2026 --device cuda

& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' train --model tide --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026' --run-id TiDE_node_shared_d512_seed2026 --device cuda

& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' train --model segrnn --input-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' --target-path 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet' --output-root 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026' --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda
```

## Linux GPU

不要猜项目路径。先进入项目 checkout 并验证：

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

### Foreground

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model dlinear --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id DLinear_node_shared_ma25_seed2026 --device cuda

python custom_models/src/benchmark_v2/run_benchmark.py train --model lightts --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id LightTS_node_shared_chunk8_seed2026 --device cuda

python custom_models/src/benchmark_v2/run_benchmark.py train --model tide --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda

python custom_models/src/benchmark_v2/run_benchmark.py train --model segrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda
```

### Foreground with unconditional shutdown

每条命令无论成功或失败都执行关机：

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model dlinear --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id DLinear_node_shared_ma25_seed2026 --device cuda; /usr/bin/shutdown -h now

python custom_models/src/benchmark_v2/run_benchmark.py train --model lightts --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id LightTS_node_shared_chunk8_seed2026 --device cuda; /usr/bin/shutdown -h now

python custom_models/src/benchmark_v2/run_benchmark.py train --model tide --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda; /usr/bin/shutdown -h now

python custom_models/src/benchmark_v2/run_benchmark.py train --model segrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda; /usr/bin/shutdown -h now
```

### nohup, exit-code preservation, sync and shutdown

```bash
nohup bash -lc 'python custom_models/src/benchmark_v2/run_benchmark.py train --model dlinear --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id DLinear_node_shared_ma25_seed2026 --device cuda; code=$?; sync; /usr/bin/shutdown -h now; exit $code' > logs/benchmark_v2/DLinear_node_shared_ma25_seed2026.log 2>&1 &

nohup bash -lc 'python custom_models/src/benchmark_v2/run_benchmark.py train --model lightts --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id LightTS_node_shared_chunk8_seed2026 --device cuda; code=$?; sync; /usr/bin/shutdown -h now; exit $code' > logs/benchmark_v2/LightTS_node_shared_chunk8_seed2026.log 2>&1 &

nohup bash -lc 'python custom_models/src/benchmark_v2/run_benchmark.py train --model tide --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda; code=$?; sync; /usr/bin/shutdown -h now; exit $code' > logs/benchmark_v2/TiDE_node_shared_d512_seed2026.log 2>&1 &

nohup bash -lc 'python custom_models/src/benchmark_v2/run_benchmark.py train --model segrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda; code=$?; sync; /usr/bin/shutdown -h now; exit $code' > logs/benchmark_v2/SegRNN_node_shared_seg2_d512_seed2026.log 2>&1 &
```

查看日志和进程：

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
```

## Safety status

以上是 E1-B 原始历史状态，保留不覆盖。E2-A 已新增下述硬件放行机制；所有正式训练仍为 `NOT_RUN`。

## RTX 4060 hardware release（E2-A 追加）

TiDE 和 SegRNN 当前均为 `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`。旧 GTX 1060 OOM 不再直接拒绝正式请求；同一条 `train` 命令会自动执行 exact preflight，preflight 子进程退出释放 CUDA context 后，自动启动新的 formal train 子进程。PASS 才进入 epoch 1；`FAIL_OOM`/`FAIL_NON_OOM` 返回非零且不创建正式 run。

独立 preflight：

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' hardware-preflight --model tide
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py' hardware-preflight --model segrnn
```

PyCharm 使用本文件既有 Interpreter、Script path、Working directory 和环境变量。Standalone Script parameters 为 `hardware-preflight --model tide` / `hardware-preflight --model segrnn`；一体化 Script parameters 使用前文 TiDE/SegRNN `train` 行，命令无需改 Registry 或二次输入。

Preflight artifacts：

- `custom_models/results_smoke/benchmark_v2/hardware_preflight/tide/8ce0530a7194465a8064/hardware_preflight.json`
- `custom_models/results_smoke/benchmark_v2/hardware_preflight/segrnn/74380f45d8773b6f9509/hardware_preflight.json`

Linux foreground：

```bash
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model tide
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model segrnn
python custom_models/src/benchmark_v2/run_benchmark.py train --model tide --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model segrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda
```

Linux nohup（每次只启动一个模型）：

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model tide --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/TiDE_node_shared_d512_seed2026.log 2>&1 &

nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model segrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/SegRNN_node_shared_seg2_d512_seed2026.log 2>&1 &
```

进程/显存/日志/exit code：

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_benchmark.py' } | Select-Object ProcessId,ParentProcessId,CommandLine
nvidia-smi
$LASTEXITCODE
```

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
nvidia-smi
wait <pid>; echo $?
sync
```

RTX 4060 preflight `NOT_RUN`；E1-B formal Full `NOT_RUN`。
