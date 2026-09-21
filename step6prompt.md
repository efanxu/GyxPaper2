# 步骤6 Prompt：Macro Prompt 与 Cross-Fusion 实验

## 0. 使用方式与研究边界

执行目录固定为 `D:\PaperProject\GyxPaper2`。本 Prompt 只实现 F0–F8 的跨粒度共享、
Macro Prompt pooling、Cross-Fusion 方向和投影共享实验；Node-Horizon Prompt/decoder
属于步骤7，MS-MG-DWU 属于步骤8。完成后写入 `HANDOFF_STEP6.md`，不自动启动正式多 seed。

## 1. 研究问题

本步骤需要区分：

1. 双分支是否需要共享信息，而不是简单并行；
2. Macro→Fine 和 Fine→Coarse 两个方向是否作用不同；
3. 双向 Cross-Fusion 的投影共享是否影响性能、参数和稳定性；
4. attention pooling 的 Macro Prompt 是否优于 mean pooling；
5. 性能变化是否能由真实 gate、attention entropy 和 representation shift 诊断支持。

诊断是模型内部关联，不是未来 horizon 的因果注意力，也不是现实反事实。任何不存在的
attention/gate 必须输出 `NOT_APPLICABLE`，不能用伪造张量填表。

## 2. 必读文件和语义审计

~~~text
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/cross_fusion.py
custom_models/src/st_mgprompt/coupling_block.py
custom_models/src/st_mgprompt/decoder.py
custom_models/src/st_mgprompt/model.py
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/empirical_protocol.py
custom_models/src/st_mgprompt/run_empirical.py
custom_models/src/st_mgprompt/diagnostics.py
custom_models/src/st_mgprompt/evaluate.py
custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md
custom_models/results/st_mgprompt_component_ablation/
custom_models/results/st_mgprompt_canonical/
~~~

先追踪实际的 Fine/Coarse 交互方向、`disable_reverse_cross`、
`disable_macro_to_fine_cross`、`macro_to_fine_mode`、`cross_fusion_recent_len`、
`macro_to_fine_exclude_recent_len` 和 `share_cross_attention_projections`。不能依据
“reverse”字面直接决定方向；必须用 forward 的 query/key/value 和 tensor shape 写一份
`F_DIRECTION_SEMANTICS.md`，在所有 F run 之前冻结。

## 3. 固定协议与 reference 映射

所有新训练变体继承 Canonical 的输入、图、扩散、decoder、loss、split、stride、batch、
seed 和评价；`protocol_profile` 默认 `STMG_FORMAL_V2`，`source_scope=internal_fusion`。
下列正式结果只读引用，不重新训练：

~~~text
F6 = A7 = Shared-Projection Cross Fusion（reference-only）
F7 = Canonical = 双向 Cross-Fusion 独立投影（reference-only）
F8 = A4 = Mean-Pooling Macro Prompt（reference-only）
A5 = Short-context Reverse Cross，作为方向/窗口诊断来源
A6 = Early-History Macro Cross Fusion，作为历史范围诊断来源
~~~

若正式 artifact 的 effective config 与上述语义不一致，先标记 `semantic_mismatch`，
不得按目录名复用。reference-only 只写 `reference.json` 和源路径，不复制 checkpoint、
metrics 或预测数组。

## 4. 变体合同

| 变体 | 定义 | 唯一差异 |
| --- | --- | --- |
| F0 | 双分支独立 | `use_macro_prompt=false` 且 `use_cross_fusion=false` |
| F1 | element-wise add | `fusion_mode=add`，输出维度固定 |
| F2 | concat+MLP | `fusion_mode=concat`，显式投影回 hidden_dim |
| F3 | unified gated fusion | 新增可审计 gate 融合，不调用双向 attention |
| F4 | Macro→Fine only | 只保留 Macro 到 Fine 的方向 |
| F5 | Fine→Coarse only | 只保留 Fine 到 Coarse 的方向 |
| F6 | 双向 Cross-Fusion，共享投影 | 只读引用 A7 |
| F7 | 双向 Cross-Fusion，独立投影 | 只读引用 Canonical |
| F8 | Mean-pooling Macro Prompt | 只读引用 A4 |

F4/F5 的具体配置字段必须根据第2节的 `F_DIRECTION_SEMANTICS.md` 对应真实方向。
如果现有字段只能表达“关闭某一方向”而命名相反，新增显式语义字段比重命名旧字段安全。

## 5. 实现任务

### 5.1 F0 独立分支

1. 保留 Fine/Coarse 编码和图/扩散，但关闭 Macro Prompt 和 Cross-Fusion 的调用路径。
2. 用模块调用计数或 forward hook 检查 Cross-Fusion、Macro Prompt 未被隐式调用。
3. 输出两个分支的 representation norm、shape 和 decoder 输入来源，证明 F0 不是“只改标签”。

### 5.2 F1/F2 基础融合

1. F1 在相同时间轴和相同 hidden_dim 上执行 element-wise add；显式处理维度和归一化。
2. F2 concat 后经过一个明确的 MLP/linear projection 回到 hidden_dim；记录层数、激活、
   dropout、参数量和初始化。
3. 两者不得调用 Cross-Attention，也不得偷偷加入 Macro token。
4. 输出融合前后 cosine similarity、representation shift 和参数量。

### 5.3 F3 unified gated fusion

1. 新增最小 `fusion_mode=unified_gated` 和 gate 参数；gate 的输入、范围、初始化、
   stop-gradient 语义必须写入配置。
2. gate 的输出必须可复现地计算均值、标准差、分位数和饱和比例。
3. gate 只能融合声明的 Fine/Coarse 表示，不得访问 target 或未来信息。
4. 记录 gate 是否按节点、时间或通道生成，并在 manifest 中固定解释层级。

### 5.4 F4/F5 双向方向

1. 保留同一 Macro Prompt 和同一分支表示，只关闭一个方向。
2. F4 只能让 Fine 从 Macro 读取；F5 只能让 Coarse 从最近 Fine 历史读取。
3. `cross_fusion_recent_len`、`macro_to_fine_exclude_recent_len`、heads 和投影维度保持
   Canonical，除非该字段就是本变体的唯一差异。
4. 保存 query/key/value 的 source branch、历史索引范围和 attention shape；审计是否存在
   最近窗口泄漏或将未来 horizon 当作输入。

### 5.5 F6/F7/F8 reference 复用

1. 运行语义审计确认 F6/F7/F8 的有效配置和结果完整性。
2. 若缺少诊断但 checkpoint 有效，执行 inference-only 导出，不改变预测。
3. 若 checkpoint 语义不符，生成 `reference_not_reusable`，不得重命名为当前 F 变体。

### 5.6 敏感性实验

对 `cross_fusion_recent_len=6/12/24/36/48`、`macro_prompt_len=1/2/4/8` 先做单 seed
筛选，但筛选只能根据验证集；关键配置再由步骤11补多 seed。每个敏感性 run 都要有
`sensitivity_parent`、验证选择规则和未选择配置，不得删除不利配置。

## 6. 诊断和测试

新增 `custom_models/tests/empirical_analysis/test_step6_fusion.py`：

1. F0 真实 forward 不调用 Macro Prompt/Cross-Fusion。
2. F1/F2 输出维度、参数量和融合算子符合合同。
3. F3 gate 值有限、范围正确、无 target/未来输入依赖。
4. F4 只存在 Macro→Fine，F5 只存在 Fine→Coarse；方向用 source branch trace 验证。
5. F6 共享投影参数确实共享，F7 投影参数独立；不能只检查布尔字段。
6. F8 pooling 实际为 mean，且 token 数、时间范围与 A4 语义一致。
7. attention 权重归一化、entropy、gate、interaction increment 都有有效值或明确
   `NOT_APPLICABLE`。
8. 所有变体输出 `[B,10,134]`，mask、clip、loss、graph、decoder 和 horizon protocol 不变。
9. 通过固定历史输入的 forward regression，诊断导出前后预测完全一致。

诊断代表样本的选择规则必须预先固定：Normal 中位波动率、高波动固定分位、ramp 最大
训练幅度且目标有效；不能为了图形美观人工挑样本。

## 7. 运行顺序和命令模板

~~~powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step6_fusion.py -q
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --dry-run
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --smoke
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --full-shape
~~~

正式训练只对未能合法复用且 smoke 通过的变体执行：

~~~powershell
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_fusion `
  --run-full --skip-completed
~~~

F6/F7/F8 默认 evaluate-only/reference-only；不得因为没有诊断就重新训练或改变正式定义。

## 8. 产物要求

~~~text
custom_models/results/empirical_analysis_v1/F/F_DIRECTION_SEMANTICS.md
custom_models/results/empirical_analysis_v1/F/F_VARIANT_MANIFEST.json
custom_models/results/empirical_analysis_v1/F/F_PROTOCOL_AUDIT.json
custom_models/results/empirical_analysis_v1/F/F_REPRESENTATION_SIMILARITY.csv
custom_models/results/empirical_analysis_v1/F/F_REPRESENTATION_SHIFT.csv
custom_models/results/empirical_analysis_v1/F/F_DIRECTION_TRACE.csv
custom_models/results/empirical_analysis_v1/F/F_GATE_STATISTICS.csv
custom_models/results/empirical_analysis_v1/F/F_ATTENTION_ENTROPY.csv
custom_models/results/empirical_analysis_v1/F/F_SENSITIVITY_SUMMARY.csv
custom_models/results/empirical_analysis_v1/F/F_PERFORMANCE_SUMMARY.csv
custom_models/results/empirical_analysis_v1/F/F_FAILURES.csv
custom_models/results/empirical_analysis_v1/F/HANDOFF_STEP6.md
~~~

每个训练 run 还必须具备 PLAN.md §2.2 的完整 artifact contract；每个诊断文件要记录
run_id、seed、horizon、scene、sample selection rule 和 `NOT_APPLICABLE` 原因。

## 9. 结果分析和验收

自动报告 H3/H6/H10 的主/辅指标、参数量、普通/高波动/ramp/困难风机结果，以及：

- F0 相对 F7 的共享收益差异；
- F1/F2/F3 的算子和容量对照；
- F4/F5 的方向边际变化；
- F6/F7 的共享投影与独立投影差异；
- F8 相对 attention pooling 的变化；
- gate/attention/representation 诊断与窗口级增益的相关关系。

步骤6只有在以下条件满足时完成：

1. F4/F5 方向语义经过代码和 trace 双重审计；
2. F6/F7/F8 正式引用来源合法且不复制指标；
3. 诊断导出不会改变预测，缺失能力正确标记 `NOT_APPLICABLE`；
4. 所有新训练变体通过唯一差异、无泄漏和 full-shape 测试；
5. 完整 manifest、失败记录和 `HANDOFF_STEP6.md` 已生成。

任何方向错标、诊断改变预测、将 Cross-Fusion 注意力当 horizon 因果或隐式改变 decoder/loss，
都必须 FAIL_CLOSED。
