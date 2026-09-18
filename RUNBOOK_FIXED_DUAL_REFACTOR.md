# ST-MGPrompt Fixed Dual 正式实验 RUNBOOK

更新时间：2026-07-25（Asia/Shanghai）

## 1. 当前可运行状态

唯一完整模型：

```text
FULL_FIXED_DUAL_DYNAMIC_MSMGDWU
```

Canonical run-id：

```text
full_fixed_dual_keep_msmgdwu_seed2026
```

引用关系：

```text
CANONICAL_FULL = P0 = A0
```

P0、A0 只生成或刷新 `reference.json`，不训练，也不复制 checkpoint。

截至本 RUNBOOK 生成时：

```text
Canonical 已迁移并验证
P0/A0 reference 已生成并验证
P1-P5 普通 smoke 已通过
A1-A7 历史普通 smoke 已通过；A6/A7 新定义需重新验证
指令要求的 12 个 full-shape preflight 已通过
P1-P5 正式长训未启动
A1-A7 正式长训未启动
```

本地显卡是 6 GiB NVIDIA GeForce GTX 1060。P2/P4/P5 的 Full-History Cross-Attn 在本地 CUDA 正式形状下显存不足；P2/P4 使用 CPU FP32 完成 preflight，P5 使用 CPU bfloat16 autocast 完成 preflight。正式训练应使用显存更充足的 Linux GPU 服务器，不得缩小正式 batch 或把 micro-batch 冒充正式形状。

## 2. 固定正式协议

```text
dataset = SDWPF
target = Patv_raw
input power = Patv_clean_for_input
mask = valid_target_mask
lookback = 144
max_pred_len = 10
eval_horizons = 3 6 10
split = 0.8 / 0.1 / 0.1（严格时间顺序）
train/val/test stride = 6 / 3 / 1
train/val/test batch = 32 / 4 / 4
epochs = 20
early stopping patience = 6
early stopping min delta = 0.01
seed = 2026
amp = enabled
windows safe mode = enabled
physical clip = [0, 1500] kW
checkpoint selection = val_official_score_h10, min
```

除 A7 外，P1-P5、A1-A6 均使用：

```text
loss = msmg_dwu_loss
granularity_weight_mode = difficulty_rate
site_weight_mode = dynamic
vadsp_gate_mode = fixed_dual
```

A7 唯一使用：

```text
loss = masked_score_aligned_hybrid
```

## 3. Windows PyCharm

### 3.1 解释器覆盖说明

原协议写的是：

```text
D:\Apps\Miniconda3\envs\env_tslib\python.exe
```

当前 Windows 推荐解释器为：

```text
D:\Apps\Miniconda3\envs\env_tslib\python.exe
```

该环境应自包含运行依赖；不要在 `PYTHONPATH` 中加入另一套 base `site-packages`。所有 Python 子进程由 runner 自动继承父进程的 `Path(sys.executable).resolve()`。

### 3.2 公共 PyCharm 配置

Interpreter：

```text
D:\Apps\Miniconda3\envs\env_tslib\python.exe
```

Working directory：

```text
D:\PaperProject\GyxPaper2
```

Environment variables：

```text
PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0
```

Windows 上 PyTorch 会提示 `expandable_segments not supported on this platform`，该提示不改变协议含义。

### 3.3 P0 reference + P1-P5 正式训练

Script path：

```text
D:\PaperProject\GyxPaper2\custom_models\src\st_mgprompt\run_precision_ablation.py
```

Script parameters：

```text
--variants P0 P1 P2 P3 P4 P5 --run-full
```

正式输出目录：

```text
D:\PaperProject\GyxPaper2\custom_models\results\st_mgprompt_precision\precision_ablation_fixed_dual_seed2026
```

预期日志状态：

```text
P0: REFERENCE_ONLY
P1: TRAIN
P2: TRAIN
P3: TRAIN
P4: TRAIN
P5: TRAIN
```

### 3.4 A0 reference + A1-A7 正式训练

Script path：

```text
D:\PaperProject\GyxPaper2\custom_models\src\st_mgprompt\run_ablation.py
```

Script parameters：

```text
--variants A0 A1 A2 A3 A4 A5 A6 A7 --run-full
```

正式输出目录：

```text
D:\PaperProject\GyxPaper2\custom_models\results\st_mgprompt_component_ablation\component_ablation_fixed_dual_seed2026
```

预期日志状态：

```text
A0: REFERENCE_ONLY
A1-A7: TRAIN
```

当前正式集合为 A0–A8。A6 为 Early-History Macro Cross Fusion（原 A6-C1），A7 为 Shared-Projection Cross Fusion（原 A6-C3），A8 为原 A7 的 `w/o MS-MG-DWU` 损失消融；A6-C2 已移除。

### 3.5 中断续跑与 checkpoint 断点续跑

正式 runner 为每个可训练变体在固定的变体输出目录中维护：

```text
last_checkpoint.pt   # 最近一个完整 epoch，供训练续跑
best_checkpoint.pt   # 当前正式评估使用的最佳 checkpoint
train_log.csv        # 已完成 epoch 的训练日志
```

每个 epoch 结束后，`last_checkpoint.pt` 会原子更新。使用 `--resume` 时，runner 会在同一个变体目录中自动读取它，并恢复模型、loss、optimizer、AMP scaler、随机数状态、early-stopping 状态和已有训练日志，从下一个 epoch 继续。若训练已经完成但评估未完成，component-ablation runner 会自动改为使用 `best_checkpoint.pt` 的 evaluate-only 路径。

断点续跑必须满足以下规则：

1. 保持原来的 `output-root`、`run-id` 和 `--variants` 不变；`--resume` 不能用来创建新的 run-id。
2. 只对当前正式定义且配置匹配的 checkpoint 使用 `--resume`。配置审计或严格 checkpoint 校验发现不匹配时，续跑会被拒绝，不得把旧语义结果当作新实验继续使用。
3. 当前正式 A4/A7 若目录中仍是旧版 Single-token Macro Prompt 或 Horizon-only Prompt 产物，必须先用下一节的 fresh Full 命令重建；重建出当前定义的 checkpoint 后，后续中断才使用 `--resume`。
4. 不要对 P0/A0 启动训练。Formal A8 Batch4 也不允许加载历史 checkpoint，继续使用 fresh Full 协议。

因此，续跑时是在原命令末尾增加：

```text
--resume
```

## 4. Windows PowerShell 等价命令

先设置环境：

```powershell
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
```

P0/P1-P5：

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_precision_ablation --variants P0 P1 P2 P3 P4 P5 --run-full
```

A0/A1-A7：

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_ablation --variants A0 A1 A2 A3 A4 A5 A6 A7 --run-full
```

仅运行当前新定义的 A6 与 A7 时，分别执行以下 fresh Full 命令（不要追加 `--resume`）：

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_ablation --variants A6 --run-full
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_ablation --variants A7 --run-full
```

当前正式 A6/A7 已经各自启动过并产生匹配的 `last_checkpoint.pt` 后，发生中断时执行以下续跑命令。命令会复用同一正式输出目录和同一固定 run-id：

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_ablation --variants A4 --run-full --resume
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_ablation --variants A7 --run-full --resume
```

若要在同一轮中续跑已经采用当前定义的 A1-A7，可使用：

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_ablation --variants A1 A2 A3 A4 A5 A6 A7 --run-full --resume --skip-completed
```

其中 `--skip-completed` 只跳过审计为 completed 的当前定义变体；不要用它掩盖 A4/A7 的旧配置不匹配。

## 5. Linux JupyterLab Terminal

当前会话只能检查本机 Windows 工作区，无法读取远端 Linux 文件系统，因此不能声称服务器路径是 `/root/workspace/GyxPaper2` 或 `/root/autodl-tmp/GyxPaper2`。先在 JupyterLab Terminal 中进入实际项目目录，再执行：

```bash
pwd
test -f custom_models/src/st_mgprompt/run_precision_ablation.py
test -f custom_models/src/st_mgprompt/run_ablation.py
export PROJECT_ROOT="$(pwd)"
```

若任一 `test -f` 失败，停止并进入正确的仓库根目录，不要猜路径。

设置环境：

```bash
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
python -c "import sys, torch, st_mgprompt; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(st_mgprompt.__file__)"
mkdir -p "$PROJECT_ROOT/custom_models/logs"
```

### 5.1 普通正式 Full 命令

P0/P1-P5：

```bash
python -m st_mgprompt.run_precision_ablation --variants P0 P1 P2 P3 P4 P5 --run-full
```

A0/A1-A7：

```bash
python -m st_mgprompt.run_ablation --variants A0 A1 A2 A3 A4 A5 A6 A7 --run-full
```

建议先完成 P0/P1-P5，再启动 A0/A1-A7。

### 5.2 Linux 中断续跑

在同一 `PROJECT_ROOT`、同一固定 run-id 下，当前正式 A6/A7 的续跑命令为：

```bash
python -m st_mgprompt.run_ablation --variants A6 --run-full --resume
python -m st_mgprompt.run_ablation --variants A7 --run-full --resume
```

不要把 `--resume` 与新的 `--run-id` 或新的 `--output-root` 组合使用。A6/A7 必须按 Windows 部分的 fresh 命令重建；Formal A8 Batch4 不使用 `--resume`。

### 5.3 无论成功或失败都自动关机

P0/P1-P5：

```bash
python -m st_mgprompt.run_precision_ablation --variants P0 P1 P2 P3 P4 P5 --run-full ; run_code=$?; printf '%s\n' "$run_code" > "$PROJECT_ROOT/custom_models/logs/precision_fixed_dual.exit_code"; sync; /usr/bin/shutdown -h now
```

A0/A1-A7：

```bash
python -m st_mgprompt.run_ablation --variants A0 A1 A2 A3 A4 A5 A6 A7 --run-full ; run_code=$?; printf '%s\n' "$run_code" > "$PROJECT_ROOT/custom_models/logs/component_fixed_dual.exit_code"; sync; /usr/bin/shutdown -h now
```

这里使用分号语义；Python 成功或失败都会记录 exit code、执行 `sync`，然后调用 `/usr/bin/shutdown -h now`。不要改为仅成功时才执行的 `&& shutdown`。

### 5.4 `nohup bash -lc` 后台运行并自动关机

先确保 `PROJECT_ROOT` 已 `export`。

P0/P1-P5：

```bash
nohup bash -lc 'cd "$PROJECT_ROOT"; export PYTHONPATH="$PROJECT_ROOT/custom_models/src${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0; python -m st_mgprompt.run_precision_ablation --variants P0 P1 P2 P3 P4 P5 --run-full ; run_code=$?; printf "%s\n" "$run_code" > "$PROJECT_ROOT/custom_models/logs/precision_fixed_dual.exit_code"; sync; /usr/bin/shutdown -h now' > "$PROJECT_ROOT/custom_models/logs/precision_fixed_dual.nohup.log" 2>&1 &
```

A0/A1-A7：

```bash
nohup bash -lc 'cd "$PROJECT_ROOT"; export PYTHONPATH="$PROJECT_ROOT/custom_models/src${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0; python -m st_mgprompt.run_ablation --variants A0 A1 A2 A3 A4 A5 A6 A7 --run-full ; run_code=$?; printf "%s\n" "$run_code" > "$PROJECT_ROOT/custom_models/logs/component_fixed_dual.exit_code"; sync; /usr/bin/shutdown -h now' > "$PROJECT_ROOT/custom_models/logs/component_fixed_dual.nohup.log" 2>&1 &
```

### 5.5 日志、进程与 exit code

查看 P0/P1-P5 日志：

```bash
tail -f "$PROJECT_ROOT/custom_models/logs/precision_fixed_dual.nohup.log"
```

查看 A0/A1-A7 日志：

```bash
tail -f "$PROJECT_ROOT/custom_models/logs/component_fixed_dual.nohup.log"
```

查看进程：

```bash
ps -ef | grep '[s]t_mgprompt.run_precision_ablation'
ps -ef | grep '[s]t_mgprompt.run_ablation'
```

查看 exit code：

```bash
cat "$PROJECT_ROOT/custom_models/logs/precision_fixed_dual.exit_code"
cat "$PROJECT_ROOT/custom_models/logs/component_fixed_dual.exit_code"
```

## 6. 正式运行顺序

```text
1. 解析 Canonical Full
2. 创建/刷新 P0 reference
3. 训练 P1
4. 训练 P2
5. 训练 P3
6. 训练 P4
7. 训练 P5
8. 汇总 P0-P5
9. 创建/刷新 A0 reference
10. 训练 A1
11. 训练 A2
12. 训练 A3
13. 训练 A4
14. 训练 A5
15. 训练 A6
16. 训练 A7
17. 汇总 A0-A8
```

不要重新训练 Canonical Full、P0 或 A0。

## 7. 运行前检查

```bash
python -m compileall -q "$PROJECT_ROOT/custom_models/src/st_mgprompt"
python -m unittest st_mgprompt.test_component_ablation
python -m st_mgprompt.run_precision_ablation --variants P0 P1 P2 P3 P4 P5 --dry-run
python -m st_mgprompt.run_ablation --variants A0 A1 A2 A3 A4 A5 A6 A7 --dry-run
```

确认：

```text
P0/A0 = REFERENCE_ONLY
其余变体 = DRY_RUN
unexpected_effective_diff_count = 0
正式目录中的旧 A6/A7 checkpoint 或 metrics 不会通过当前 config identity 审计
```
