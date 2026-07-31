# E1-B formal runbook（RTX 4060 硬件预检放行版）

本文件只准备正式运行命令；E1-B 尚未执行任何正式训练。

当前模型状态应为：

- DLinear：`AVAILABLE_TRAINABLE`
- LightTS：`AVAILABLE_TRAINABLE`
- TiDE：`AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`
- SegRNN：`AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`

其中 TiDE、SegRNN 在旧的 6GB GPU 上 exact full-shape smoke 发生过 OOM，但其源码、协议、普通 smoke、真实数据 smoke、forward/backward、checkpoint strict reload 和 artifact validation 均已通过。

旧机器的 OOM 记录必须永久保留：

```text
local_full_shape_status = FAIL_OOM
non_oom_engineering_checks = PASS
formal_hardware_preflight_required = true
supports_train = true
```

在按修订版 E2-A 完成 Registry/runtime/CLI 修改后，TiDE、SegRNN 的正式 `train` 命令不会因旧机器的 OOM 记录而直接被拒绝。正式 `train` 请求应自动执行：

```text
检查匹配的 hardware preflight
-> 若不存在 PASS 结果，启动独立 preflight 子进程
-> 使用精确正式 shape 做 AMP forward + masked loss + backward
-> preflight 子进程退出并释放 CUDA context
-> PASS 后自动启动新的正式训练子进程
-> 进入 epoch 1
```

如果 RTX 4060 上 preflight 仍 OOM，则停止，不进入 epoch 1，不自动缩小 batch、节点数、lookback、horizon 或模型容量。

> 重要：在 Codex 尚未完成上述 E2-A 硬件放行修改前，TiDE、SegRNN 仍可能被当前 Registry 拦截。请先完成并验收修订版 E2-A，再执行本文件中的 TiDE、SegRNN 正式命令。

正式输出根：

```text
custom_models/results/benchmark_v2/basic_lightweight_seed2026
```

Run IDs：

- `DLinear_node_shared_ma25_seed2026`
- `LightTS_node_shared_chunk8_seed2026`
- `TiDE_node_shared_d512_seed2026`
- `SegRNN_node_shared_seg2_d512_seed2026`

---

## 1. Windows PyCharm

四个配置的共同设置：

- Interpreter：`D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script path：`D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory：`D:\PaperProject\GyxPaper2`
- Environment variables：

```text
PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0
```

### 1.1 DLinear

Script parameters：

```text
train --model dlinear --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id DLinear_node_shared_ma25_seed2026 --device cuda
```

执行行为：直接启动正式训练。

### 1.2 LightTS

Script parameters：

```text
train --model lightts --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id LightTS_node_shared_chunk8_seed2026 --device cuda
```

执行行为：直接启动正式训练。

### 1.3 TiDE

Script parameters：

```text
train --model tide --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda
```

执行行为：

```text
自动检查匹配的 full-shape preflight
-> 没有 PASS 记录时自动启动独立 preflight 子进程
-> preflight PASS 后自动启动新的正式训练进程
```

不需要手工修改 Registry，也不需要绕过安全检查。

### 1.4 SegRNN

Script parameters：

```text
train --model segrnn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda
```

执行行为：

```text
自动检查匹配的 full-shape preflight
-> 没有 PASS 记录时自动启动独立 preflight 子进程
-> preflight PASS 后自动启动新的正式训练进程
```

不需要手工修改 Registry，也不需要绕过安全检查。

---

## 2. Windows PowerShell

先进入项目根目录并设置环境：

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'

$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES='0'

$PY='D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$RUNNER='D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py'
$INPUT='D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet'
$TARGET='D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet'
$OUT='D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026'
```

### 2.1 DLinear

```powershell
& $PY $RUNNER train `
  --model dlinear `
  --input-path $INPUT `
  --target-path $TARGET `
  --output-root $OUT `
  --run-id 'DLinear_node_shared_ma25_seed2026' `
  --device cuda

$exitCode = $LASTEXITCODE
Write-Host "DLinear finished, exit code: $exitCode"
exit $exitCode
```

### 2.2 LightTS

```powershell
& $PY $RUNNER train `
  --model lightts `
  --input-path $INPUT `
  --target-path $TARGET `
  --output-root $OUT `
  --run-id 'LightTS_node_shared_chunk8_seed2026' `
  --device cuda

$exitCode = $LASTEXITCODE
Write-Host "LightTS finished, exit code: $exitCode"
exit $exitCode
```

### 2.3 TiDE

TiDE 使用同一条正式 `train` 命令。完成修订版 E2-A 后，该命令会自动执行硬件 preflight，并在 PASS 后继续正式训练。

```powershell
& $PY $RUNNER train `
  --model tide `
  --input-path $INPUT `
  --target-path $TARGET `
  --output-root $OUT `
  --run-id 'TiDE_node_shared_d512_seed2026' `
  --device cuda

$exitCode = $LASTEXITCODE
Write-Host "TiDE preflight/train finished, exit code: $exitCode"
exit $exitCode
```

### 2.4 SegRNN

SegRNN 使用同一条正式 `train` 命令。完成修订版 E2-A 后，该命令会自动执行硬件 preflight，并在 PASS 后继续正式训练。

```powershell
& $PY $RUNNER train `
  --model segrnn `
  --input-path $INPUT `
  --target-path $TARGET `
  --output-root $OUT `
  --run-id 'SegRNN_node_shared_seg2_d512_seed2026' `
  --device cuda

$exitCode = $LASTEXITCODE
Write-Host "SegRNN preflight/train finished, exit code: $exitCode"
exit $exitCode
```

---

## 3. Windows 顺序运行四个模型

正式批量运行时必须保证：

```text
一个模型 = 一个独立 Python 进程
```

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'

$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES='0'

$PY='D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$RUNNER='D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py'
$INPUT='D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet'
$TARGET='D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet'
$OUT='D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\basic_lightweight_seed2026'

$MODELS = @(
    @{ Id='dlinear'; RunId='DLinear_node_shared_ma25_seed2026' },
    @{ Id='lightts'; RunId='LightTS_node_shared_chunk8_seed2026' },
    @{ Id='tide'; RunId='TiDE_node_shared_d512_seed2026' },
    @{ Id='segrnn'; RunId='SegRNN_node_shared_seg2_d512_seed2026' }
)

foreach ($item in $MODELS) {
    Write-Host "Starting $($item.Id)..." -ForegroundColor Cyan

    & $PY $RUNNER train `
      --model $item.Id `
      --input-path $INPUT `
      --target-path $TARGET `
      --output-root $OUT `
      --run-id $item.RunId `
      --device cuda

    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Error "$($item.Id) failed, exit code: $exitCode. Stop the suite."
        exit $exitCode
    }

    Write-Host "$($item.Id) completed. Its Python process has exited." -ForegroundColor Green
}

Write-Host 'All E1-B models completed.' -ForegroundColor Green
exit 0
```

该脚本不会同时加载四个模型，也不会让 Registry 持有 GPU 模型实例。

---

## 4. Linux GPU

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

### 4.1 Foreground

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model dlinear --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id DLinear_node_shared_ma25_seed2026 --device cuda
```

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model lightts --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id LightTS_node_shared_chunk8_seed2026 --device cuda
```

TiDE：同一条正式命令自动执行 preflight，PASS 后继续训练：

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model tide --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda
```

SegRNN：同一条正式命令自动执行 preflight，PASS 后继续训练：

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model segrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda
```

---

## 5. Linux：单个模型运行结束后无条件关机

### DLinear

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model dlinear --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id DLinear_node_shared_ma25_seed2026 --device cuda; /usr/bin/shutdown -h now
```

### LightTS

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model lightts --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id LightTS_node_shared_chunk8_seed2026 --device cuda; /usr/bin/shutdown -h now
```

### TiDE

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model tide --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id TiDE_node_shared_d512_seed2026 --device cuda; /usr/bin/shutdown -h now
```

### SegRNN

```bash
python custom_models/src/benchmark_v2/run_benchmark.py train --model segrnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 --run-id SegRNN_node_shared_seg2_d512_seed2026 --device cuda; /usr/bin/shutdown -h now
```

---

## 6. Linux nohup：日志、exit code、sync 和自动关机

### DLinear

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train \
  --model dlinear \
  --input-path dataset/sdwpf_model_input_base.parquet \
  --target-path dataset/sdwpf_eval_target.parquet \
  --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 \
  --run-id DLinear_node_shared_ma25_seed2026 \
  --device cuda
code=$?
sync
/usr/bin/shutdown -h now
exit $code
' > logs/benchmark_v2/DLinear_node_shared_ma25_seed2026.log 2>&1 &
```

### LightTS

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train \
  --model lightts \
  --input-path dataset/sdwpf_model_input_base.parquet \
  --target-path dataset/sdwpf_eval_target.parquet \
  --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 \
  --run-id LightTS_node_shared_chunk8_seed2026 \
  --device cuda
code=$?
sync
/usr/bin/shutdown -h now
exit $code
' > logs/benchmark_v2/LightTS_node_shared_chunk8_seed2026.log 2>&1 &
```

### TiDE

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train \
  --model tide \
  --input-path dataset/sdwpf_model_input_base.parquet \
  --target-path dataset/sdwpf_eval_target.parquet \
  --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 \
  --run-id TiDE_node_shared_d512_seed2026 \
  --device cuda
code=$?
sync
/usr/bin/shutdown -h now
exit $code
' > logs/benchmark_v2/TiDE_node_shared_d512_seed2026.log 2>&1 &
```

### SegRNN

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train \
  --model segrnn \
  --input-path dataset/sdwpf_model_input_base.parquet \
  --target-path dataset/sdwpf_eval_target.parquet \
  --output-root custom_models/results/benchmark_v2/basic_lightweight_seed2026 \
  --run-id SegRNN_node_shared_seg2_d512_seed2026 \
  --device cuda
code=$?
sync
/usr/bin/shutdown -h now
exit $code
' > logs/benchmark_v2/SegRNN_node_shared_seg2_d512_seed2026.log 2>&1 &
```

---

## 7. 日志、进程和显存查看

查看某个模型日志：

```bash
tail -f logs/benchmark_v2/<run-id>.log
```

查看 benchmark 进程：

```bash
ps -ef | grep '[r]un_benchmark.py'
```

查看 GPU：

```bash
watch -n 2 nvidia-smi
```

Windows PowerShell 查看 Python 进程：

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -match '^python(.exe)?$' -and
    $_.CommandLine -match 'benchmark_v2'
  } |
  Select-Object ProcessId, Name, CommandLine
```

Windows 查看 GPU：

```powershell
nvidia-smi -l 2
```

---

## 8. Preflight 结果要求

TiDE、SegRNN 的正式训练前，自动 preflight 必须使用：

```text
B=32
T=144
N=134
C=16
H=10
AMP=true
forward=true
masked loss=true
backward=true
```

必须使用完整正式配置。

TiDE：

```text
d_model=512
d_ff=2048
e_layers=2
d_layers=1
c_out=1
```

SegRNN：

```text
seg_len=2
d_model=512
lookback=144
pred_len=10
```

不得自动减小 batch、节点、lookback、horizon、d_model 或层数，也不得自动启用 node chunk 或梯度累积。

Preflight PASS 后，preflight 进程必须退出，再启动新的正式训练进程。

Preflight FAIL_OOM 时：

```text
不进入 epoch 1
不创建看似完成的正式 run
写入明确失败记录
返回非零 exit code
```

---

## 9. Safety status

截至本 Runbook 更新时：

```text
E1-B 正式 DLinear：NOT_RUN
E1-B 正式 LightTS：NOT_RUN
E1-B 正式 TiDE：NOT_RUN
E1-B 正式 SegRNN：NOT_RUN
RTX 4060 hardware preflight：NOT_RUN
```

本 Runbook 中 TiDE、SegRNN 的命令只有在修订版 E2-A 已完成并通过以下验收后才能执行：

```text
Registry 状态：
AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED

supports_train = true
formal_hardware_preflight_required = true
local_full_shape_status = FAIL_OOM
non_oom_engineering_checks = PASS

train 请求：
不会因旧 OOM 记录直接拒绝
会自动运行独立 preflight
preflight PASS 后自动继续正式训练
```
