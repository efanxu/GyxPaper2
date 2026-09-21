# 步骤4 Prompt：双语义图与先验约束实验

## 0. 使用方式与边界

执行目录固定为 `D:\PaperProject\GyxPaper2`。本 Prompt 只实现 G0–G8 的图先验、图分支
分配、自适应支持集和随机重连对照；扩散方向/阶数属于步骤5，Prompt/Cross-Fusion 属于
步骤6，不能在本阶段顺手修改。完成后生成 `HANDOFF_STEP4.md`，不自动启动多 seed 长训。

## 1. 研究问题

本步骤要验证：

1. Fine 分支与 Micro 图、Coarse 分支与 Macro 图的 matched assignment 是否比共享或交换图更符合数据统计；
2. 距离先验与训练区间统计相似性是否提供互补信息；
3. 先验约束的自适应边权是否比自由候选图更稳定；
4. 同等稀疏度的随机重连是否能复现正式图的效果。

只能将结果表述为“统计语义匹配”“先验支持集约束”或“模型内部边权稳定性”。
不得把邻接边直接解释为尾流传播、风向因果或地理因果。

## 2. 必读文件和当前实现审计

~~~text
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/data.py
custom_models/src/st_mgprompt/graph_prior.py
custom_models/src/st_mgprompt/graph_layers.py
custom_models/src/st_mgprompt/model.py
custom_models/src/st_mgprompt/coupling_block.py
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/empirical_protocol.py
custom_models/src/st_mgprompt/run_empirical.py
custom_models/src/st_mgprompt/diagnostics.py
custom_models/src/st_mgprompt/metrics.py
dataset/sdwpf_turb_location_elevation.csv
custom_models/graphs/trend_prior_v4_2/
RUNBOOK_FIXED_DUAL_REFACTOR.md
~~~

先定位 Micro/Macro/distance 图的真正构造函数、训练区间边界、TopK、坐标列、是否使用
elevation、self-loop 和自适应图的支持 mask。不要使用目录名或旧报告中的图名称推断
实际矩阵。图身份审计使用节点顺序、节点数、坐标来源、边数、TopK、构造参数和度统计，
不生成文件哈希。

## 3. 固定协议和图身份

所有 G 变体使用 T0/Canonical 的相同数据、输入、split、stride、batch、loss、decoder、
hidden_dim、coupling、扩散阶数和评价指标。默认 profile 是 `STMG_FORMAL_V2`；每个 run
都必须写入 `protocol_profile=EMPIRICAL_ANALYSIS_V1`、`source_scope=internal_mechanism`
和 `paired_reference=G0`。

训练区间之外的数据不得参与：

~~~text
统计相似性计算
距离图阈值或 TopK 选择
自适应候选支持集
图标准化参数
场景/难度分组阈值
~~~

`graph_identity.json` 至少包含：`node_count`、`node_order_source`、`coordinate_columns`、
`location_file`、`distance_definition`、`micro_top_k`、`macro_top_k`、`self_loop`、
`normalization`、`elevation_used`、`micro_source`、`macro_source`、`fit_split`、
`edge_count`、`row_degree_summary` 和 `not_applicable_reason`（无图模型时）。

## 4. 变体合同

| 变体 | 定义 | 唯一差异 |
| --- | --- | --- |
| G0 | Micro/Fine + Macro/Coarse matched | 只读引用 Canonical |
| G1 | 两分支共享 Micro 图 | `branch_graph_assignment=shared_micro` |
| G2 | 两分支共享 Macro 图 | `branch_graph_assignment=shared_macro` |
| G3 | Micro/Macro 互换 | `branch_graph_assignment=swapped` |
| G4 | 仅 distance 图 | `graph_prior_component=distance_only` |
| G5 | 仅统计相似图 | `graph_prior_component=statistics_only` |
| G6 | 自由自适应候选 | `adaptive_support_mode=free` 或等价关闭 prior support |
| G7 | 固定双图 | `adaptive_support_mode=fixed` 或关闭可学习边权 |
| G8 | 保持每行边数的随机重连 | `graph_rewire_mode=degree_preserving_random` |

如果当前 `STMGPromptConfig` 尚无上述字段，只能新增最小字段、校验值和序列化支持；
不能把多个含义塞进一个布尔值，也不能只改 manifest 标签。`empirical_protocol.py`
中的 `expected_unique_diff` 必须与 forward 中真实生效的字段一致。

## 5. 实现任务

### 5.1 G0 基线和图快照

1. 从 Canonical 的正式 effective config 重新构造 G0 图，不复制旧 `.npy` 后再改名。
2. 对 Micro、Macro、distance 三类矩阵保存稀疏摘要、行度、非零值分位数和节点顺序。
3. 用单元测试确认 G0 的图来源、TopK、坐标列和分支 assignment 与 Canonical 一致。
4. 对已有图快照只读复用；若训练区间或图语义不一致，标记 `requires_rebuild`。

### 5.2 G1–G3 分支 assignment

1. 在模型 forward 中显式打印/记录 Fine 与 Coarse 分支实际收到的图 ID。
2. G1 两分支都使用同一 Micro 图；G2 两分支都使用同一 Macro 图；G3 交换 matched assignment。
3. 节点数、边数、图层参数和自适应权重预算保持不变；不要因共享矩阵而悄悄改变层数。
4. 输出 `branch_graph_assignment_trace.json`，说明每个 branch、每个层和每次 forward 的图来源。

### 5.3 G4–G5 图来源隔离

1. G4 只保留正式 distance 核和预先冻结的 TopK；不得混入 trend similarity。
2. G5 只保留训练区间统计相似性；不得使用测试期相关性或未来窗口。
3. 在图构造器中返回 `source_components`，并在 `protocol_check.json` 验证来源集合恰好符合变体合同。
4. 输出两种图与 G0 的 support overlap 和 edge-distance 分布；结论只描述统计差异。

### 5.4 G6–G7 自适应支持集

1. G6 允许所有非对角节点作为候选，但保持相同的 temperature、edge parameterization、
   hidden_dim、图层数和训练预算。
2. G7 保留固定 prior 图，不创建或更新可学习边权；运行时检查 adaptive 参数没有梯度和更新。
3. 对 G6 记录 prior support 外新增的边、边长、度分布、稀疏度和跨 seed 稳定性。
4. 不能把 G6 的自由候选称为“真实图”，只能称为 free adaptive control。

### 5.5 G8 随机重连

1. 以 G0 每行非零边数为目标，在同一节点集合内重连；保存随机种子和重连算法。
2. 明确是否保留 self-loop、对称性和有向性；默认不新增未声明 self-loop。
3. 每行度数、总边数和图张量形状必须与参考图一致；若无法保持，run fail closed。
4. G8 只能作为结构对照，不替换 Canonical 图，也不进入 G0 的 reference artifact。

## 6. 必须实现的测试

新增 `custom_models/tests/empirical_analysis/test_step4_graph_semantics.py`：

1. G0 图矩阵和 branch assignment 与正式 Canonical manifest 一致。
2. G1/G2/G3 的实际输入图 ID 与变体合同一致，不能只改变标签。
3. G4 的 source_components 不含统计图；G5 不含 distance 图。
4. G6 的支持集允许先验之外候选，G7 的 adaptive 参数不存在梯度更新。
5. G8 每行非零边数、节点数、self-loop 和方向性符合合同。
6. 所有图的生成输入只来自训练区间、坐标和已冻结公共元数据。
7. 对输入未来时间点或验证/测试数据的修改不会改变训练期图。
8. 所有变体输出 `[B,10,134]`，图矩阵有限且无未声明 NaN/Inf。
9. `empirical_config_diff` 不允许 batch、seed、mask、clip、loss、decoder 等冻结字段改变。

## 7. 运行顺序和命令模板

~~~powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step4_graph_semantics.py -q
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --dry-run
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --smoke
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --full-shape
~~~

当前 G1–G5/G8 若仍为 `pending`，先完成配置、图构造器和测试，再运行 smoke；不能使用
`--continue-on-error` 把 pending 当成 completed。用户批准后才执行：

~~~powershell
python -m st_mgprompt.run_empirical --family G --variants G1 G2 G3 G4 G5 G6 G7 G8 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_graph `
  --run-full --skip-completed
~~~

## 8. 产物和汇总

~~~text
custom_models/results/empirical_analysis_v1/G/G_VARIANT_MANIFEST.json
custom_models/results/empirical_analysis_v1/G/G_PROTOCOL_AUDIT.json
custom_models/results/empirical_analysis_v1/G/G_GRAPH_IDENTITY.csv
custom_models/results/empirical_analysis_v1/G/G_BRANCH_ASSIGNMENT_TRACE.json
custom_models/results/empirical_analysis_v1/G/G_SUPPORT_OVERLAP.csv
custom_models/results/empirical_analysis_v1/G/G_EDGE_DISTANCE_SUMMARY.csv
custom_models/results/empirical_analysis_v1/G/G_NODE_GROUP_METRICS.csv
custom_models/results/empirical_analysis_v1/G/G_FAILURES.csv
custom_models/results/empirical_analysis_v1/G/HANDOFF_STEP4.md
~~~

每个 run 还要保存 prior/effective/final adjacency 的可复核摘要、模型参数量、逐 horizon
metrics、逐风机 metrics 和 `prediction_metadata.json`。不能保存巨大张量而不提供来源和
抽样规则；也不能用图字段为无图外部模型伪造结果。

## 9. 结果分析与验收

自动汇总 G0–G8 的 H3/H6/H10 Score、MAE、RMSE、R²、逐步/逐风机误差、图稀疏度和参数量，
并在公共分组冻结后报告高/低空间依赖风机、High-volatility、Ramp 和 Localized disturbance。
报告 semantic-fit 与性能的方向性关系时使用相关/分组描述，不报告因果系数。

步骤4只有在以下条件满足时完成：

1. G0–G8 的实际唯一差异和图来源均可审计；
2. G0 图身份、节点顺序和训练区间来源完整；
3. G3 确实交换图，G6/G7 的 adaptive 行为符合合同，G8 度数约束通过；
4. 全部正式 run 有 profile/source_scope/artifact contract，失败项有 failure report；
5. `HANDOFF_STEP4.md` 记录未完成变体和下一步。

任意未来信息进入图构造、图标签与实际输入不一致、G7 仍更新边权或 G8 改变度数时，
必须 FAIL_CLOSED，并从图构造审计重新开始。
