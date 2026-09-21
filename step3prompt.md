# 步骤3 Prompt：Fixed Dual 与同时间轴设计实验

## 0. 使用方式

把本文件整体交给负责步骤3的实现代理或研究协作者。执行目录固定为
`D:\PaperProject\GyxPaper2`。本 Prompt 只负责 T0–T5 的时间粒度实验，不实现步骤4–8
的图、扩散、Cross-Fusion、解码器或损失新机制。完成后必须生成独立的
`HANDOFF_STEP3.md`，并等待用户决定是否启动正式长训。

## 1. 研究问题与证据目标

本步骤要回答三个可区分的问题：

1. Fine 局部变化和 Coarse 长期趋势是否分别提供预测信息？
2. 将两种粒度放在同一历史时间轴上，是否优于先降采样再恢复的非同轴表示？
3. 双粒度的收益是否只是参数增多，还是来自受控的时间结构和定向交互？

结论只能写成“时间表示与误差表现相符”“支持同时间轴设计”或“与该机制一致”。
不得把 T4 的性能差异写成物理因果，也不得预先假定 T0 在所有指标上最优。

## 2. 必须先读取的文件

实现前完整阅读并在 `HANDOFF_STEP3.md` 中记录版本和实际使用入口：

~~~text
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/data.py
custom_models/src/st_mgprompt/model.py
custom_models/src/st_mgprompt/coupling_block.py
custom_models/src/st_mgprompt/decoder.py
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/empirical_protocol.py
custom_models/src/st_mgprompt/run_empirical.py
custom_models/src/st_mgprompt/evaluate.py
custom_models/src/st_mgprompt/metrics.py
custom_models/src/st_mgprompt/diagnostics.py
custom_models/src/st_mgprompt/test_empirical_protocol.py
custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md
RUNBOOK_FIXED_DUAL_REFACTOR.md
~~~

先查清 `vadsp_gate_mode`、Fine/Coarse 分支、`coarse_windows=[6,18,36]`、
`use_stmg_coupling_block` 和 Canonical 的真实调用路径。不要仅按配置字段名称判断
实现是否存在；必须追踪到 forward 中实际使用的位置。

## 3. 不可改变的协议

所有 T 变体默认使用 `STMG_FORMAL_V2`/`EMPIRICAL_ANALYSIS_V1` 的 Canonical 配对协议：

~~~text
dataset = SDWPF
target = Patv_raw
input power = Patv_clean_for_input
mask = valid_target_mask
lookback = 144
max_pred_len = 10
eval_horizons = [3, 6, 10]
split = strict chronological [0.8, 0.1, 0.1]
sample stride = train/val/test = 6/3/1
batch = 32/4/4（除非 protocol_profile 明确声明其他 profile）
seed = 2026（多 seed 由步骤11执行）
AMP、epochs、patience、min_delta、physical clip、checkpoint selection 均继承 Canonical
output = [B, 10, N]，N=134
~~~

必须把 `protocol_profile`、`source_scope`、实际 batch、唯一配置差异写入：
`resolved_config.json`、`effective_config.json`、`protocol_check.json`、
`source_scope.json` 和结果 manifest。禁止通过降低 batch、改变 stride、缩短 lookback、
改变损失或改变 clip 来“修复”单个变体的显存问题。

## 4. 变体和唯一差异合同

| 变体 | 研究含义 | 唯一允许的结构差异 | 状态 |
| --- | --- | --- | --- |
| T0 | Canonical Fixed Dual | 只读引用 `CANONICAL_FULL` | reference-only |
| T1 | Fine-only | `vadsp_gate_mode=fine_only`，保留其余结构 | 可训练 |
| T2 | Coarse-only | `vadsp_gate_mode=coarse_only`，保留其余结构 | 可训练 |
| T3 | Shared temporal transform dual | 两分支共享声明的时间变换参数，维度和输出预算固定 | 需实现 |
| T4 | Downsample-upsample Coarse dual | 仅将 Coarse 分支替换为因果聚合-恢复 | 需实现 |
| T5 | Direct concat/MLP | 关闭 Macro Prompt/Cross-Fusion，改为容量匹配的 concat+MLP | 需实现 |

T0 不重新训练、不复制 checkpoint、不手工复制 metrics。T1/T2 只能改变 gate；如果
实现需要新增字段，必须先更新 `STMGPromptConfig` 的校验和 `empirical_protocol.py` 的
`expected_unique_diff`，再运行唯一差异审计。T3–T5 尚未实现时必须保留
`implementation_status=pending`，不能把注册成功写成已完成。

## 5. 实现任务

### 5.1 建立基线快照

1. 由 `canonical_config()` 生成 T0 reference 描述，记录 Canonical run-id、模型摘要、
   10 步输出形状、参数量和协议字段。
2. 对 T1/T2 执行 dry-run，确认 `actual_diff_fields` 只包含
   `vadsp_gate_mode`；若出现 batch、seed、loss、graph、decoder 或 feature 差异，
   立即 `FAIL_CLOSED`。
3. 将每个变体的 `paired_reference=T0`、`profile` 和 `source_scope` 写入
   `T_VARIANT_MANIFEST.json`。

### 5.2 实现 T1/T2

1. 复用现有 VADSP/Fine/Coarse 分支，不复制模型类。
2. `fine_only` 必须停用 Coarse 分支的贡献；`coarse_only` 必须停用 Fine 分支的贡献。
3. 保持输入仍是完整 144 步历史，不能通过裁剪历史伪造单粒度模型。
4. 记录每个分支的激活比例、表示范数和输出形状，以便排查“关闭分支但仍在计算”的情况。
5. 训练 loss、图、扩散、coupling、decoder 和 checkpoint selection 均继承 T0。

### 5.3 实现 T3

1. 新增最小字段，例如 `temporal_branch_transform="shared"`；名称必须与实际实现一致。
2. Fine/Coarse 使用同一组时间变换参数，但允许输入窗口或通道投影存在必要的形状适配。
3. 若共享参数导致参数量下降，增加一个明确记录的输出投影，而不是隐式增加隐藏层。
4. `model_summary.json` 必须同时给出总参数、可训练参数、共享参数数量和两个分支输出维度。
5. 单元测试要检查两分支确实引用同一参数对象或等价共享状态，而不是仅仅把字段命名为 shared。

### 5.4 实现 T4：因果 Downsample-upsample 对照

T4 只能作为时间对齐的受控反例。实现时遵守：

1. Coarse 聚合窗口只能使用当前历史位置及其之前的数据。
2. 如果步长为 `s`，在位置 `t` 的聚合输入必须满足 `index <= t`；不能先用完整 144 步
   做全局池化后再插值。
3. 恢复到 144 个位置时，为每个恢复位置保存 `source_interval_start`、
   `source_interval_end`、`causal_cutoff` 和 `upsample_rule`。
4. 明确处理不能整除的尾部：只能使用向前截断、因果复制或左侧填充，并将规则写入配置，
   禁止使用未来位置补齐。
5. 产出 `coarse_alignment_trace.csv`，至少包含样本、节点、输出位置和访问上界；
   正式运行可只保存聚合摘要，但 smoke 必须保存完整 trace。

### 5.5 实现 T5：直接拼接对照

1. 关闭 Macro Prompt 和双向 Cross-Fusion，但保留 Fine/Coarse 两个分支。
2. 将两分支表示对齐到相同时间位置后 concat，再经过一个显式 MLP 投影到 Canonical
   的 hidden_dim。
3. MLP 层数、激活、dropout 和输出维度写入配置；参数量与 T0 的差异必须在报告中透明列出。
4. 不得把 T5 命名成“无交互”后仍保留隐藏的 Cross-Fusion 或 gate；执行 forward hook/模块
   统计确认相关模块未被调用。

### 5.6 注册、运行和状态管理

1. 更新 `empirical_protocol.py` 的 T3/T4/T5 状态、配置差异和说明。
2. 更新 `run_empirical.py` 的 family/variant 选择，不修改 P/A/E runner。
3. smoke、full-shape 和 formal 结果必须写入不同目录；建议：

~~~text
custom_models/results/empirical_analysis_v1/T/smoke/<variant>/
custom_models/results/empirical_analysis_v1/T/full_shape/<variant>/
custom_models/results/empirical_analysis_v1/T/seed_2026/<variant>/
~~~

## 6. 必须实现的测试

新增或扩展 `custom_models/tests/empirical_analysis/test_step3_fixed_dual.py`：

1. T0 的 Canonical 配置和 graph/loss/decoder identity 与正式 manifest 一致。
2. T1/T2 只有预期 gate 字段变化。
3. T3 的两分支时间变换参数共享且输出形状一致。
4. T4 的每个 Coarse 位置访问上界不超过当前位置；随机修改上界之后的数据不会改变该位置。
5. T4 的未来目标 `y` 永远不会进入输入或对齐索引。
6. T5 的 Macro Prompt/Cross-Fusion forward 路径未被调用，concat+MLP 输出维度正确。
7. T0–T5 输出均为有限值 `[B,10,134]`，mask 全无效批次不会产生 NaN。
8. 所有配置的冻结字段与 Canonical 完全相同，包含 profile、seed、stride、clip 和 metrics。

因果测试要使用合成数据和真实小 batch 两套输入；不能只检查配置字符串。

## 7. 运行顺序与命令模板

先执行静态检查，不启动长训：

~~~powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step3_fixed_dual.py -q
python -m st_mgprompt.run_empirical --family T --variants T0 T1 T2 T3 T4 T5 --dry-run
python -m st_mgprompt.run_empirical --family T --variants T0 T1 T2 T3 T4 T5 --smoke
python -m st_mgprompt.run_empirical --family T --variants T0 T1 T2 T3 T4 T5 --full-shape
~~~

如果当前 runner 尚未实现 `--full-shape`，先补齐 CLI 和独立输出模式，再继续；不得把
`--smoke` 结果写入正式目录。正式运行命令只能在用户批准后执行：

~~~powershell
python -m st_mgprompt.run_empirical --family T --variants T1 T2 T3 T4 T5 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_mechanism `
  --run-full --skip-completed
~~~

完成单 seed 全部审计后，步骤11再补充多 seed；步骤3 Prompt 不自行启动五 seed 长训。

## 8. 必须生成的产物

~~~text
custom_models/results/empirical_analysis_v1/T/T_VARIANT_MANIFEST.json
custom_models/results/empirical_analysis_v1/T/T_PROTOCOL_AUDIT.json
custom_models/results/empirical_analysis_v1/T/T_UNIQUE_DIFF_MATRIX.csv
custom_models/results/empirical_analysis_v1/T/T_ALIGNMENT_SUMMARY.csv
custom_models/results/empirical_analysis_v1/T/T_PERFORMANCE_SUMMARY.csv
custom_models/results/empirical_analysis_v1/T/T_DIFFICULTY_GROUPS.csv
custom_models/results/empirical_analysis_v1/T/T_FAILURES.csv
custom_models/results/empirical_analysis_v1/T/HANDOFF_STEP3.md
~~~

每个 completed run 还必须具有 PLAN.md §2.2 的完整 artifact contract。T0 只写
`reference.json` 和可追溯来源，不复制 checkpoint/metrics。

## 9. 结果分析要求

自动生成 H3/H6/H10 的 Score、MAE、RMSE、R²、逐步误差、逐风机误差、参数量和有效样本数。
同时输出 Normal、High-volatility、Ramp-up、Ramp-down 和 Shared difficult Top10%（若
步骤9的公共分组已冻结）结果。额外报告：

- T1/T2 随 horizon 的退化方向；
- T4 的峰值时间偏差、事件窗口重构误差和访问对齐统计；
- T3/T5 的参数差异与性能差异；
- 同一测试窗口、同一 mask 下的配对增益。

结果汇总必须保留 partial/failed/reference-only，缺失值不得填 0。正文只在唯一差异审计
通过且 run_status=completed 时使用。

## 10. 验收与失败处理

只有以下条件全部满足才将步骤3标记为 complete：

1. T1–T5 的唯一差异审计通过；
2. T4 因果访问测试通过；
3. T5 确实没有隐藏 Cross-Fusion；
4. 所有正式产物包含 profile、source_scope、mask、prediction_start_index 和 checkpoint 信息；
5. 至少完成 seed=2026 的可审计结果，或为每个未完成变体生成 failure_report；
6. `HANDOFF_STEP3.md` 说明实际运行命令、测试结果和未完成事项。

任意未来泄漏、profile 改写、未声明配置差异或指标手工复制都必须 FAIL_CLOSED；修复后
重新从 dry-run 开始，不能沿用受污染的 checkpoint。
