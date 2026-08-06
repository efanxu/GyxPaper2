# Python 解释器路径全项目审计

审计日期：2026-07-26（Asia/Shanghai）  
项目根目录：`D:\PaperProject\GyxPaper2`

## 结论（修改前）

- 指定文本扩展名共扫描 3,146 个文件：`*.py`、`*.ps1`、`*.bat`、`*.cmd`、`*.json`、`*.yaml`、`*.yml`、`*.toml`、`*.ini`、`*.env`、`*.md`。
- 旧解释器字面量 `D:\Apps\Miniconda\envs\env_tslib\python.exe`、反斜杠转义变体和正斜杠变体：未在当前 UTF-8 文本扫描中命中。
- 实际执行代码仍存在一处同样不应存在的硬编码当前 Windows 解释器：`custom_models/src/st_mgprompt/formal_runner.py:32,41`。
- 实际启动 Python 子进程的代码仅发现 `formal_runner.py:199` 的 `subprocess.run()`；其命令源头使用 `args.python`，因此会把预生成或人工传入的解释器路径带入子进程。
- `--python` 在项目运行时代码中仅发现于 `formal_runner.py:41`，默认值为硬编码绝对路径。
- 未发现 `PYTHON_EXECUTABLE`、`PYTHON_PATH`、`PYTHON_BIN`、`python_executable` 或 `python_path` 作为解释器选择配置的其它执行代码命中。
- 当前运行文档 `HANDOFF_FIXED_DUAL_REFACTOR.md` 和 `RUNBOOK_FIXED_DUAL_REFACTOR.md` 仍描述跨环境借用 `D:\Apps\Miniconda3\Lib\site-packages`；这不是当前 `env_tslib` 的独立运行配置，应移除。

## 审计明细

| 文件路径 | 行号 | 发现的解释器路径或相关代码 | 分类 | 是否影响实际执行 | 计划处理方式 |
|---|---:|---|---|---|---|
| `custom_models/src/st_mgprompt/formal_runner.py` | 32 | `AUTHORIZED_PYTHON = r"D:\Apps\Miniconda3\envs\env_tslib\python.exe"` | A. 实际执行代码 | 是 | 删除硬编码；统一使用 `Path(sys.executable).resolve()`。 |
| `custom_models/src/st_mgprompt/formal_runner.py` | 41 | `parser.add_argument("--python", default=AUTHORIZED_PYTHON)` | A. 实际执行代码 | 是 | 默认改为当前进程解释器；显式覆盖须存在且必须与当前解释器一致，避免静默跨环境。 |
| `custom_models/src/st_mgprompt/formal_runner.py` | 199 | `subprocess.run(command, cwd=PROJECT_ROOT, check=False)` | A. 实际执行代码 | 是 | 启动前调用防御性 `normalize_python_command()`，并打印父/子解释器及完整命令。 |
| `custom_models/src/st_mgprompt/run_precision_ablation.py` | — | 委托 `formal_runner.run_family("precision")` | A. 实际执行代码 | 间接 | 由 `formal_runner` 集中修复；不改变 P0-P5 矩阵。 |
| `custom_models/src/st_mgprompt/run_ablation.py` | — | 委托 `formal_runner.run_family("component_ablation")` | A. 实际执行代码 | 间接 | 由 `formal_runner` 集中修复；不改动 A0-A8 协议。 |
| `custom_models/src/st_mgprompt/run_st_mgprompt.py` | 518 | `sys.version` 环境记录 | B. 测试或 smoke 代码 | 否 | 保留；由父/子进程实际启动日志补充解释器字段。 |
| `custom_models/src/st_mgprompt/full_shape_matrix_smoke.py` | — | 未发现 `subprocess`、`--python` 或 Miniconda 路径 | B. 测试或 smoke 代码 | 否 | 保留现有进程内 full-shape smoke；验证由 runner 统一负责。 |
| `Custom Model 协议.md` | 17, 23, 24, 29, 33 | 当前 Windows 解释器示例及 Miniconda3 说明 | D. 当前运行文档 | 仅影响用户手工启动 | 保留当前解释器示例，明确通用 Python 代码不得依赖绝对路径。 |
| `HANDOFF_FIXED_DUAL_REFACTOR.md` | 55, 61, 64, 67, 73 | 当前解释器与 `D:\Apps\Miniconda3\Lib\site-packages` 跨环境依赖说明 | D. 当前运行文档 | 影响用户配置 | 保留当前解释器，删除 base `site-packages` 借用说明，改为环境自包含。 |
| `RUNBOOK_FIXED_DUAL_REFACTOR.md` | 86, 92, 95, 102, 114, 196, 207, 213 | PyCharm/PowerShell 当前解释器及跨环境 `PYTHONPATH` | D. 当前运行文档 | 影响用户配置 | Windows 示例固定为当前推荐解释器；`PYTHONPATH` 只保留项目源码目录。 |

## F. Canonical provenance / manifest / record 记录

以下命中来自已生成的结果、命令快照、状态快照、preflight 或 full-shape 报告，均不是 Python 解释器选择源，也不会被运行代码读取来决定 `subprocess.run()` 的 `command[0]`。按照用户要求保留原始证据，不做全局替换：

- `custom_models/results/st_mgprompt_component_ablation/component_ablation_fixed_dual_seed2026/A1-A8/run_command.json`（A1-A8 各一份）。
- `custom_models/results/st_mgprompt_component_ablation/component_ablation_p1_seed2026/component_ablation_commands.json`。
- `custom_models/results/st_mgprompt_component_ablation/component_ablation_p1_seed2026/component_ablation_status.json`。
- `custom_models/results/st_mgprompt_precision/precision_ablation_fair_main_seed2026_all_p0_p5/precision_ablation_status.json`。
- `custom_models/results/st_mgprompt_precision/precision_ablation_fixed_dual_seed2026/P1-P5/run_command.json`。
- `custom_models/results/st_mgprompt_full_candidate/**/candidate_preflight_report.json`、`full_shape_smoke_report.json`。
- `custom_models/results/st_mgprompt_oom_fix_smoke/**/precision_ablation_{failure_summary,status}.json`。
- `custom_models/results/st_mgprompt_supplementary_ablation_smoke/{scheduler_smoke_seed2026,synthetic_smoke_seed2026}/supplementary_ablation_{commands,status}.json`。
- `custom_models/results_smoke/st_mgprompt_fixed_dual_refactor_full_shape/**/full_shape_smoke_{report,summary,failure}.json`。

这些文件记录的是历史运行时环境或已生成命令；不能作为未来运行配置。Canonical checkpoint、原始 config、原始 metrics、canonical manifest、provenance、artifact record、删除清单和删除状态均不在本次路径修复的修改范围内。

## E. 纯历史记录

本次扫描未发现旧路径字面量，但上述历史结果中的解释器路径仍属于历史/派生证据。若未来发现旧 `Miniconda` 路径出现在 provenance、status、command snapshot 或 handoff 的历史段落中，应保留原值，并新增当前推荐值与 `inherit_sys_executable` 政策字段，而不是伪装改写历史。

## G. 删除清单

以下文件明确列为保护对象，本次不修改、不删除：

- `DELETE_MANIFEST_refactor.json`
- `DELETE_MANIFEST_refactor_bytecode.json`
- `DELETE_STATUS_refactor.json`

本任务没有任何删除操作；不涉及模型结果目录清理。

## 修复计划

1. 在 `formal_runner.py` 增加 `get_current_python_executable()` 与 `normalize_python_command()`。
2. 命令构造和 `subprocess.run()` 前均以 `Path(sys.executable).resolve()` 为准，并传递 `os.environ.copy()`。
3. 记录 `Parent Python executable`、`Child Python executable`、`sys.prefix`、Python 版本、Torch 版本、CUDA 状态和 GPU 名称。
4. 删除当前运行文档中跨环境 base `site-packages` 配置。
5. 只执行 compileall、import、dry-run 和短 smoke/子进程启动检查，不启动正式长训。
