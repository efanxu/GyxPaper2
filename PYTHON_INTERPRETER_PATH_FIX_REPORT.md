# Python 解释器路径永久修复报告

日期：2026-07-26（Asia/Shanghai）  
项目：`D:\PaperProject\GyxPaper2`

## 1. 根因

`formal_runner.py` 的子进程命令此前由 `--python`/硬编码路径决定，导致父 runner 与 P1-P5/A1-A8 子 runner 可能使用不同解释器。预生成 JSON 中的解释器路径也可能被误认为当前运行配置。修复后，命令构造和 `subprocess.run()` 前均以当前父进程的 `Path(sys.executable).resolve()` 为准。

## 2. 搜索范围

扫描项目内 3,146 个 UTF-8 文本文件，扩展名包括：

`*.py`、`*.ps1`、`*.bat`、`*.cmd`、`*.json`、`*.yaml`、`*.yml`、`*.toml`、`*.ini`、`*.env`、`*.md`。

重点检查了 formal runner、precision/component runner、`run_st_mgprompt.py`、协议/registry/config、smoke/full-shape/preflight、测试、RUNBOOK/HANDOFF、命令 JSON、provenance/manifest/record 和删除清单。

完整修改前审计见：[PYTHON_INTERPRETER_PATH_AUDIT.md](D:\PaperProject\GyxPaper2\PYTHON_INTERPRETER_PATH_AUDIT.md)。

## 3. 发现与分类

### A. 实际执行代码

- `custom_models/src/st_mgprompt/formal_runner.py`：原有硬编码当前 Windows 路径、`--python` 默认值和唯一 `subprocess.run()`。
- `run_precision_ablation.py` 与 `run_ablation.py` 通过 `formal_runner.run_family()` 间接使用该子进程逻辑。

### B. 测试或 smoke 代码

- 未发现其它 Python 子进程启动器。
- `full_shape_matrix_smoke.py` 是进程内 full-shape smoke，没有解释器硬编码。
- `run_st_mgprompt.py` 增加子进程运行时环境日志：`Child Python`、`sys.prefix`、Python version、Torch version、CUDA available、GPU name。

### C. Windows 运行脚本

- 未发现 `.ps1`、`.bat`、`.cmd` 中的旧解释器路径。
- 当前 Windows 命令示例集中在 RUNBOOK/HANDOFF 文档中，统一保留当前推荐路径。

### D. 当前运行文档

已修改：

- `Custom Model 协议.md`
- `HANDOFF_FIXED_DUAL_REFACTOR.md`
- `RUNBOOK_FIXED_DUAL_REFACTOR.md`

Windows 当前示例使用 `D:\Apps\Miniconda3\envs\env_tslib\python.exe`；PyCharm/PowerShell 的 `PYTHONPATH` 只包含项目源码目录，不再加入另一套 base `site-packages`。

### E/F. 历史记录与 Canonical provenance/manifest/record

已生成的 result command/status/preflight/full-shape JSON 保留原始环境证据，不被运行代码读取为未来解释器配置。以下保护对象未修改：

- Canonical checkpoint、原始 config、原始 metrics、canonical manifest、provenance、artifact records。
- `DELETE_MANIFEST_refactor.json`
- `DELETE_MANIFEST_refactor_bytecode.json`
- `DELETE_STATUS_refactor.json`

### G. 删除清单

本任务未删除任何文件，也未修改任何正式结果目录；没有执行批量删除命令。

## 4. 实际修改文件

- `custom_models/src/st_mgprompt/formal_runner.py`
  - 新增 `get_current_python_executable()`。
  - 新增 `normalize_python_command()`。
  - `--python` 默认值改为 `sys.executable`。
  - 显式 `--python` 必须存在且必须解析到当前父解释器，否则立即失败。
  - 子进程启动前打印父/子解释器和完整命令。
  - `subprocess.run()` 使用 `cwd=str(PROJECT_ROOT)`、`env=os.environ.copy()`。
- `custom_models/src/st_mgprompt/run_st_mgprompt.py`
  - 环境摘要增加 `sys_prefix`。
  - 子 runner 打印完整运行时环境摘要。
- `Custom Model 协议.md`
- `HANDOFF_FIXED_DUAL_REFACTOR.md`
- `RUNBOOK_FIXED_DUAL_REFACTOR.md`
- `PYTHON_INTERPRETER_PATH_AUDIT.md`（审计交付物）
- `PYTHON_INTERPRETER_PATH_FIX_REPORT.md`（本报告）

未修改 `config.py`、`experiment_protocol.py`、`registry.py` 的实验协议、变体矩阵、seed、batch、loss、checkpoint 选择或输出协议。

## 5. 子进程策略

实际执行路径为：

```text
parent runner sys.executable
    -> Path(sys.executable).resolve()
    -> normalize_python_command(command)
    -> command[0] == Path(sys.executable).resolve()
    -> subprocess.run(..., env=os.environ.copy())
```

因此 Linux 使用 `python -m st_mgprompt.run_precision_ablation ...` 时，子进程自动使用 Linux 当前解释器；Windows 绝对路径只存在于 Windows 用户运行文档/配置示例，不进入通用执行代码。

## 6. 字符串残留扫描

- 旧解释器路径 `D:\Apps\Miniconda\envs\env_tslib\python.exe`：实际执行代码、测试代码、当前 Windows 配置和当前运行文档中为 0。
- 扫描中保留的命中只出现在审计/修复报告作为“被审计旧路径”文本，不参与执行。
- `D:\Apps\Miniconda\Lib\site-packages`：执行代码和当前 Windows 配置中为 0。
- `D:\Apps\Miniconda3\Lib\site-packages`：当前 RUNBOOK/HANDOFF 配置中为 0；仅审计报告保留说明性历史文本。
- 非结果 Python 源码中唯一的 `subprocess` 启动点仍是修复后的 `formal_runner.py:256`。

## 7. 验证结果

### 7.1 目标 Windows 解释器环境

目标文件存在，并报告：

```text
sys.executable = D:\Apps\Miniconda3\envs\env_tslib\python.exe
sys.prefix = D:\Apps\Miniconda3\envs\env_tslib
purelib = D:\Apps\Miniconda3\envs\env_tslib\Lib\site-packages
```

但该环境当前实际缺少 `numpy` 和 `torch`：

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'numpy'
```

因此没有把 base `site-packages` 重新加入配置，也没有安装、复制或改写环境包；目标环境的 Torch import 和真实训练验收被依赖缺失阻断。

### 7.2 compileall

使用目标解释器执行 `custom_models/src/st_mgprompt` compileall：

```text
PASS
```

### 7.3 import

使用目标解释器执行 `import st_mgprompt`：

```text
BLOCKED: target env_tslib lacks numpy
```

### 7.4 dry-run

使用依赖完整的 `D:\Apps\Miniconda3\python.exe` 仅验证动态命令构造，输出目录为 `custom_models/results_smoke/interpreter_path_fix_dry_run`：

```text
P0: REFERENCE_ONLY
P1-P5: DRY_RUN
A0: REFERENCE_ONLY
A1-A8: DRY_RUN
```

两组命令均 exit 0，未启动正式训练。生成命令的 `command[0]` 为实际启动该验证父进程的 `sys.executable`。

### 7.5 P1/A1 子进程真实启动

使用依赖完整的 Miniconda base 仅做短 smoke 链路验证：

```text
P1: SMOKE_COMPLETED, returncode=0
A1: SMOKE_COMPLETED, returncode=0, protocol_passed=true
```

日志均明确显示：

```text
Parent Python: D:\Apps\Miniconda3\python.exe
Child Python: D:\Apps\Miniconda3\python.exe
```

P1 full-shape 子进程也成功进入 child runner 并打印同样的父/子解释器、Torch、CUDA、GPU 信息；该机器在 full-shape 计算阶段未产生最终 summary，未出现 `FileNotFoundError: [WinError 2]`。这项 full-shape 运行结果不冒充完整通过。

### 7.6 Canonical/协议回归

已确认：

```text
canonical_verification.passed = true
state_dict_strict_load = true
loss_state_dict_strict_load = true
parameter_count = 278,534
A0/P0 -> 同一 Canonical artifact
P0/A0 -> REFERENCE_ONLY
```

Canonical content record 保持：

```text
best_checkpoint.pt = <removed-content-record>
config.json        = <removed-content-record>
metrics.csv        = <removed-content-record>
```

Canonical 文件未被本次修复触碰。

## 8. 最终结论

```text
Actual runtime code stale interpreter references: 0

Current Windows scripts stale interpreter references: 0

Current recommended Windows interpreter:
D:\Apps\Miniconda3\envs\env_tslib\python.exe

Child process policy:
inherit Path(sys.executable).resolve()

Canonical artifact changed:
false
```

唯一剩余前置事项是为 `env_tslib` 安装其自身缺失的 `numpy`、`torch` 等依赖；这不属于本次解释器路径修复，且本次没有跨环境借包或修改环境。
