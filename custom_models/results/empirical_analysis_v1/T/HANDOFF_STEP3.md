# 步骤 3 交接：Fixed Dual 与同轴对齐实证协议

- 完成日期：2026-09-21（Asia/Shanghai）
- 协议档案：`STMG_FORMAL_V2`
- 来源范围：`internal_mechanism`
- 结果根目录：`custom_models/results/empirical_analysis_v1/T`

## 完成状态

步骤 3 的协议固化、T0–T5 变体实现、审计、smoke 验证、真实维度显存预检、测试和交付表均已完成。正式长训练没有启动；T1–T5 的正式目录以 `failure_report.json` 明确记录 `not_started`，不会把缺失指标写成 0。

T0 仅引用规范运行 `full_fixed_dual_keep_msmgdwu_seed2026`，没有复制或伪造指标。T1–T5 各自只改变一个预先声明的机制字段：

| 变体 | 唯一变化 | 实现状态 |
|---|---|---|
| T0 | 规范 Fixed Dual 引用 | reference-only |
| T1 | `vadsp_gate_mode=fine_only` | ready |
| T2 | `vadsp_gate_mode=coarse_only` | ready |
| T3 | `temporal_branch_transform=shared` | ready |
| T4 | `coarse_alignment_mode=causal_downsample_upsample` | ready |
| T5 | `cross_granularity_interaction=direct_concat_mlp` | ready |

## 关键实现

- T1/T2：非活动分支不执行时序编码，贡献张量严格为 0；smoke 诊断记录分支激活比例、范数和形状。
- T3：两分支复用同一个 `causal_tcn` 参数对象，同时保留恒等初始化的分支输出投影；模型摘要记录共享参数量。
- T4：按窗口 `[6, 18, 36]` 执行因果降采样与 `causal_repeat` 上采样。smoke 生成 1,152 条 `coarse_alignment_trace.csv`，全部满足访问上界不晚于当前截止时刻。
- T5：不实例化 Macro Prompt 或 Symmetric Cross Fusion，改用 `2D → 432 → D` 的两层拼接 MLP；参数量 278,709，相对 T0 的 278,534 增加 175（小于 0.1%）。
- 全尺寸检查直接读取规范检查点中的 `A_macro_prior` 和 `A_micro_prior`，两者均为 134×134；不依赖工作区中的调试图文件。

## 验证结果

- Python 编译检查：通过。
- 回归测试：`42 passed`；另有 1 条既有 Matplotlib `labels` 参数弃用警告。
- 协议 dry-run：T0–T5 全部通过；没有缺失、意外或冻结字段变化。
- smoke：T1–T5 全部通过，产物契约完整。
- 全尺寸预检：T1–T5 均完成前向、损失和反向，输入 `[32,144,134,16]`，输出 `[32,10,134]`，损失和梯度均有限。

| 变体 | CUDA 峰值显存 MiB |
|---|---:|
| T1 | 3913.79 |
| T2 | 4820.16 |
| T3 | 8356.08 |
| T4 | 9312.83 |
| T5 | 8709.24 |

验证命令使用解释器 `D:\Apps\Miniconda3\envs\env_tslib\python.exe`，并设置 `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src`。

## 主要代码改动

- `custom_models/src/st_mgprompt/config.py`
- `custom_models/src/st_mgprompt/volatility_patching.py`
- `custom_models/src/st_mgprompt/coupling_block.py`
- `custom_models/src/st_mgprompt/model.py`
- `custom_models/src/st_mgprompt/diagnostics.py`
- `custom_models/src/st_mgprompt/empirical_protocol.py`
- `custom_models/src/st_mgprompt/run_empirical.py`
- `custom_models/src/st_mgprompt/run_st_mgprompt.py`
- `custom_models/src/st_mgprompt/step3_reporting.py`
- `custom_models/tests/empirical_analysis/test_step3_fixed_dual.py`

## 只读依据

- `C:/Users/12811/Desktop/步骤3prompt.md`
- `C:/Users/12811/Desktop/PLAN.md`
- `C:/Users/12811/Desktop/实证分析.docx`
- `custom_models/docs/A0_A8_FORMAL_DEFINITIONS.md`
- `custom_models/docs/RUNBOOK_FIXED_DUAL_REFACTOR.md`
- 现有数据、评估、指标、解码器及规范检查点。

## 正式训练入口

正式 seed=2026 长训练需要用户明确授权。授权后从项目根目录执行：

```powershell
$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' -m st_mgprompt.run_empirical --family T --variants T1 T2 T3 T4 T5 --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_mechanism --run-full --skip-completed
```

训练完成后重新生成汇总，填充 T1–T5 的性能、困难组和失败统计；当前空值表示尚无正式预测，而不是 0。
