# ST-MGPrompt 实证分析完整实施计划（审查修订版）

## 1 计划目标

本计划用于在现有 GyxPaper2 项目中完整实现 ST-MGPrompt 的实证分析。最终目标不是简单增加若干消融实验，而是形成一条能够回答“为什么需要该模型、各模块为什么这样设计、模型在什么条件下有效、结论是否稳定”的证据链。

最终应形成以下成果：

1. 数据层证据：证明短时扰动、长期趋势、空间语义和预测难度异质性真实存在。
2. 模型层证据：证明 Fixed Dual、双语义图、先验约束自适应边权、二阶双向扩散、Macro Prompt、双向 Cross-Fusion、Node-Horizon Prompt、last-state 解码和 MS-MG-DWU 各自对应明确问题。
3. 场景层证据：说明模型在普通、高波动、爬坡、空间一致变化和局部扰动场景中的表现。
4. 统计层证据：通过多随机种子、配对分块 bootstrap 和 DM 检验验证结论稳定性。
5. 解释层证据：输出图结构、Prompt、门控、节点和视界权重等内部诊断。
6. 论文层成果：生成可直接写入论文的表格、图片、结果摘要和章节草稿。

## 2 当前项目事实与不可改变的正式协议

项目根目录：

~~~text
D:\PaperProject\GyxPaper2
~~~

当前核心实现：

~~~text
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/data.py
custom_models/src/st_mgprompt/model.py
custom_models/src/st_mgprompt/graph_prior.py
custom_models/src/st_mgprompt/graph_layers.py
custom_models/src/st_mgprompt/cross_fusion.py
custom_models/src/st_mgprompt/coupling_block.py
custom_models/src/st_mgprompt/decoder.py
custom_models/src/st_mgprompt/losses.py
custom_models/src/st_mgprompt/metrics.py
custom_models/src/st_mgprompt/evaluate.py
custom_models/src/st_mgprompt/diagnostics.py
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/run_st_mgprompt.py
~~~

正式实验身份必须保持：

~~~text
CANONICAL_FULL = P0 = A0
Canonical run-id = full_fixed_dual_keep_msmgdwu_seed2026
P0-P5、A0-A8 的现有编号、定义、checkpoint 和结果不得重写
~~~

冻结的公共协议：

~~~text
dataset = SDWPF
target = Patv_raw
input power = Patv_clean_for_input
mask = valid_target_mask
lookback = 144
max_pred_len = 10
eval_horizons = [3, 6, 10]
split = [0.8, 0.1, 0.1]，严格时间顺序
train/val/test stride = 6/3/1
physical clip = [0, 1500] kW
checkpoint selection = validation official Score at H10，越低越好
输入 scaler、target scaler、图先验和场景阈值只能在训练区间拟合
~~~

正式论文指标：

~~~text
主指标：official Score
辅助指标：MAE、RMSE、R²
H3、H6、H10 均由同一个 10 步输出截取前缀
平均 Score 只作描述性汇总，不替代分视界结果
~~~

### 2.1 当前正式协议与实验 profile

当前协议只保留本章实证分析需要的正式 ST-MGPrompt profile。每个 run 必须把
`protocol_profile`、`source_scope`、实际 batch 写入 `effective_config.json`、
`protocol_check.json`、`prediction_metadata.json` 和汇总 manifest：

| profile_id | 适用范围 | train/val/test batch | 比较边界 |
| --- | --- | --- | --- |
| `STMG_FORMAL_V2` | Canonical、P0-P5、A0-A7 及本计划 T/G/D/F/N/L/R 变体 | 32/4/4 | 只与同 profile 做组件比较 |
| `STMG_A8_BATCH4_V1` | 当前 A8 的专用 `uniform_train_batch4_v1` fresh run | 4/4/4 | 只作为独立 A8 scope，不与 32/4/4 结果做组件因果比较 |
| `EMPIRICAL_ANALYSIS_V1` | 本计划 T/G/D/F/N/L/R 新实验 | 继承配对参考 profile | 必须由 `protocol_profile` 审计后比较 |

优先级是：实际有效配置和代码检查器 > 当前 RUNBOOK/正式 contract > 本计划文字。
当前正式结果根目录：Canonical 在 `custom_models/results/st_mgprompt_canonical/`，
P0-P5 在 `custom_models/results/st_mgprompt_precision/`，A0-A7 在
`custom_models/results/st_mgprompt_component_ablation/`；只读 alias 只能保存
`reference.json`，不得复制 checkpoint 或手工重写 metrics。

当前 A0-A8 的唯一来源是 `custom_models/src/st_mgprompt/experiment_protocol.py`
及 `custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md`：

~~~text
A0 Canonical Full（仅引用 CANONICAL_FULL）
A1 w/o Spatial Graph
A2 w/o Adaptive Graph
A3 w/o Diffusion
A4 Mean-Pooling Macro Prompt
A5 Short-context Reverse Cross
A6 Early-History Macro Cross Fusion
A7 Shared-Projection Cross Fusion
A8 w/o MS-MG-DWU（独立 STMG_A8_BATCH4_V1 scope）
~~~

不得依据其他章节的旧 A6/A7/A8 名称、旧 checkpoint 或旧结果目录推断当前语义。
当前 A8 contract 的正式 run-id/path 由 `a8_batch4_contract.py` 提供；默认审计入口为
`custom_models/results/st_mgprompt_uniform_bs4/component_ablation_a8_bs4_seed2026/`，
不得把它移动到 Canonical/A0 的 32/4/4 目录或覆盖已有 A8 reference。

### 2.2 统一 artifact contract

任何被称为 `formal`、`completed` 或进入论文表格的 run，必须至少包含以下文件；
缺失时只能标记为 `partial` 或 `failed`，不得由汇总器补零：

~~~text
resolved_config.json       # 命令行与默认值解析后的配置
effective_config.json      # 真正传给 dataloader/model/loss 的配置
protocol_check.json        # profile、split、stride、batch、mask、clip、checkpoint 选择检查
source_scope.json          # run 来源、是否只读引用、可比较范围
model_summary.json         # 参数量、可训练参数、输入输出形状
best_checkpoint.pt
last_checkpoint.pt
train_log.csv
metrics_eval_h3.json
metrics_eval_h6.json
metrics_eval_h10.json
metrics.csv
prediction_metadata.json   # prediction_start_index、节点顺序、horizon、mask 语义
run_status.json             # completed/partial/failed/reference-only 及失败原因
evaluation_complete.json
failure_report.json         # 失败或不适用时必需，completed 时可写空 reasons
~~~

需要效率结论时还必须有 `efficiency.json` 或等价行级记录，其中包含
`parameter_count`、`trainable_parameter_count`、`model_file_size_bytes`、
`total_train_seconds`、`best_epoch`、`inference_latency_ms`、`throughput`、
`peak_memory_mb`、硬件/软件环境和实际 batch。图模型另写
`node_order_manifest.json`、`graph_identity.json`；使用节点顺序、节点数、坐标文件、
邻接构造规则和边统计作为身份证据，不生成文件哈希。

## 3 全局实施规则

所有后续步骤必须遵守：

1. 先读取并理解现有实现，不新建第二套训练框架。
2. 新增实验使用独立命名空间，不占用 P、A 已有编号。
3. 推荐新实验族：
   - T 系列：时间粒度和 Fixed Dual。
   - G 系列：图先验和扩散。
   - D 系列：扩散方向和阶数（与 G 分开注册，避免把图来源和传播算子混为一项）。
   - F 系列：Prompt、Cross-Fusion 和解码。
   - N 系列：Node-Horizon Prompt 与解码器。
   - L 系列：MS-MG-DWU。
   - R 系列：稳健性和困难场景。
4. 不修改或覆盖历史结果目录。新结果统一写入：

~~~text
custom_models/results/empirical_analysis_v1/
~~~

5. 数据分析产物统一写入：

~~~text
custom_models/results/empirical_analysis_v1/data_evidence/
~~~

6. 汇总表、统计检验和论文图统一写入：

~~~text
custom_models/results/empirical_analysis_v1/summary/
paper_figures/empirical_analysis_v1/
~~~

7. 每次训练必须遵守 §2.2 的 artifact contract；`active_config.json`、`config.json` 可以作为兼容别名保留，但不能替代 `resolved_config.json`、`effective_config.json`、`protocol_check.json`、`source_scope.json`、`prediction_metadata.json`、`last_checkpoint.pt` 和 `failure_report.json`。
8. 需要样本级分析的实验必须输出通过审计的预测数组、目标、掩码和 prediction_start_index。
9. 不根据预期结论修改实验，不筛掉不利结果，不伪造数值。
10. 不把统计相关性、地理邻近或模型内部注意力写成物理因果或尾流传播证明。
11. 不批量删除文件，不使用递归删除命令。需要清理时只处理一个明确文件，或保留旧产物并更换 run-id。
12. 不生成文件哈希或 SHA256。身份审计使用现有配置字段、路径、run-id、模型名和协议字段。
13. 不同证据来源分成不同 `source_scope`；相关性、地理邻近或模型内部权重不得被写成组件的纯因果证明。
14. 分析型 Prompt 只允许读取、导出、汇总和生成诊断，不自动启动正式长训；训练型 Prompt 必须先完成静态审计、普通 smoke 和 full-shape smoke。
15. 所有失败、未实现、依赖缺失和不适用模型都保留 manifest 与 `failure_report.json`，不得通过删除行、填 0 或复制其他 seed 隐藏。
16. 每个 Prompt 完成后生成阶段性 `HANDOFF_<阶段>.md`，列出修改文件、运行命令、测试结果、未完成项和下一步；长训练由用户单独决定是否启动。

## 4 完成顺序与依赖

~~~text
步骤 0 现状审计与协议冻结
  ↓
步骤 1 建立增量实验框架
  ↓
步骤 2 数据事实分析
  ├── 步骤 3 Fixed Dual
  ├── 步骤 4 双语义图
  ├── 步骤 5 扩散机制
  ├── 步骤 6 Cross-Fusion
  ├── 步骤 7 Node-Horizon Prompt 与解码
  └── 步骤 8 MS-MG-DWU
          ↓
步骤 9 困难场景与稳健性
步骤 10 内部可解释性
步骤 11 多随机种子与统计检验
步骤 12 效率审计
          ↓
步骤 13 总表和论文图
步骤 14 论文实证章节
步骤 15 最终一致性审计
~~~

---

## 步骤 0 现有资产与协议审计

### 目标

建立可复用资产清单，确认哪些结果可以直接使用、哪些必须重新推理、哪些必须重新训练，避免重复计算或混用旧语义结果。

### 必须输出

~~~text
custom_models/results/empirical_analysis_v1/audit/existing_runs.csv
custom_models/results/empirical_analysis_v1/audit/existing_artifacts.json
custom_models/results/empirical_analysis_v1/audit/reuse_matrix.csv
custom_models/results/empirical_analysis_v1/audit/audit_report.md
~~~

### 实现 Prompt

~~~text
你正在 D:\PaperProject\GyxPaper2 中执行 ST-MGPrompt 实证分析的资产审计。

先完整阅读 README.md、RUNBOOK_FIXED_DUAL_REFACTOR.md、HANDOFF_FIXED_DUAL_REFACTOR.md、
custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md，
以及以下源码：
config.py、experiment_protocol.py、run_st_mgprompt.py、evaluate.py、metrics.py、
diagnostics.py、scripts/export_formal_predictions.py。

只做读取、审计和新增报告，不修改任何现有 checkpoint、结果和正式定义。

递归检查 custom_models/results 下与以下对象有关的运行：
Canonical、P0-P5、A0-A8，以及本计划 T/G/D/F/N/L/R 变体。

对每个运行记录：
experiment_family、variant、run_id、model_name、seed、配置路径、checkpoint 状态、
train_complete、evaluation_complete、run_status、H3/H6/H10 指标、参数量、
是否有 metrics_per_step.csv、metrics_per_turbine.csv、metrics_by_volatility.csv、
图快照、Prompt/Cross-Fusion 诊断、loss_weights.csv、site_weight_final.csv、
正式预测导出和 prediction_start_index、protocol_profile、source_scope、
train/val/test batch、loss identity、checkpoint selection 和模型文件大小。

必须区分：
1. 可直接复用的正式数值；
2. checkpoint 完整但缺少预测导出，可 evaluate-only 或 inference-only 补充；
3. 配置语义不匹配，必须重新训练；
4. 状态冲突，仅可保留精度指标、不可用于效率结论；
5. 完全缺失。

生成 existing_runs.csv、existing_artifacts.json、reuse_matrix.csv 和 audit_report.md。
reuse_matrix.csv 至少包含：证据任务、所需变体、现有资产、可否复用、缺失项、下一步。

不得生成哈希，不得删除或改名任何历史文件。完成后运行最小 JSON/CSV 可读性检查，
并在 audit_report.md 中明确列出可立即开始的无训练分析。
~~~

### 验收标准

- P0/A0被识别为同一Canonical引用，而不是两次独立训练。
- A6、A7、A8采用当前正式定义。
- P4/P5状态冲突被明确标注，未用于效率结论。
- 所有可复用资产均能追溯到具体路径；profile 不匹配的资产不会被误标为组件因果证据。

---

## 步骤 1 建立增量实验框架

### 目标

在不改变现有正式实验身份的前提下，为 T、G、F、L、R 系列建立可审计、可恢复、可汇总的统一运行框架。

### 推荐新增文件

~~~text
custom_models/src/st_mgprompt/empirical_protocol.py
custom_models/src/st_mgprompt/run_empirical.py
custom_models/src/st_mgprompt/summarize_empirical.py
custom_models/src/st_mgprompt/test_empirical_protocol.py
~~~

### 实现 Prompt

~~~text
在现有 st_mgprompt 包中增量实现 empirical_analysis_v1 实验框架。

禁止修改 P0-P5、A0-A8 的定义、run-id 和历史结果目录。
复用 STMGPromptConfig、make_dataloaders、build_model、训练器、evaluate_model、
checkpoint 恢复、配置审计和结果汇总逻辑。

新增 empirical_protocol.py：
1. 定义 EmpiricalVariant 数据结构；
2. 每个变体必须有 variant_id、display_name、family、base_config、
   config_overrides、expected_unique_diff、trainable、paired_reference；
3. 注册 T、G、D、F、N、L、R 变体；
4. 提供 apply_empirical_variant、assert_empirical_expected_diff、
   write_empirical_matrix；
5. 对所有变体执行唯一差异审计，禁止隐式改变 batch、seed、split、stride、
   lookback、max_pred_len、物理裁剪和评价指标。

新增 run_empirical.py：
1. 支持 --family、--variants、--seeds、--profile、--source-scope、--dry-run、
   --smoke、--full-shape、--run-full、--resume、--skip-completed、--evaluate-only；
2. 输出目录固定为 custom_models/results/empirical_analysis_v1/<family>/<run_id>/；
3. 多种子目录采用 seed_<seed>/model_name；
4. 复用现有原子状态写入和 checkpoint 逻辑；
5. dry-run 打印每个变体相对基准的唯一配置差异；
6. reference-only 变体只生成 reference.json；
7. 所有训练运行输出 §2.2 的完整 artifact contract，并在 run manifest 中写入
   protocol_profile、source_scope、paired_reference、actual batch 和配置差异；
8. profile 不匹配时 fail closed；不得用未声明 profile 静默改变 batch、loss 或评价协议。

新增 summarize_empirical.py：
1. 汇总所有 seed 和 horizon；
2. 输出 long-format 与 wide-format CSV；
3. 不把缺失运行填成 0；
4. 标记 completed、partial、failed、reference-only；
5. 输出 mean、std、有效 seed 数和原始单 seed 值。

补充单元测试：
1. 每个变体只有声明过的配置差异；
2. 输出形状保持 [B,10,134]；
3. 不读取未来观测；
4. scaler、图和阈值只用训练区间；
5. P/A 系列定义未变化；
6. smoke 前向、loss backward 和评价均为有限值。

先运行 compileall、现有 st_mgprompt 核心测试和新测试，再执行所有变体 dry-run、普通
smoke 和 full-shape smoke。不要启动长训，直到三者全部通过；分析型调用默认只生成
报告和 evaluate-only 入口，不自动执行正式长训。
~~~

### 验收标准

- 新框架能运行单变体、变体组和多随机种子。
- 原正式注册表无任何语义变化。
- 每个新变体都有严格的唯一差异报告。

---

## 步骤 2 数据事实与模型动机分析

### 目标

在训练数据上证明模型所针对的四类问题：时间尺度差异、同轴对齐需要、空间语义差异、节点和视界难度异质性。

### 推荐新增文件

~~~text
scripts/empirical_analysis/build_data_evidence.py
scripts/empirical_analysis/plot_data_evidence.py
custom_models/tests/empirical_analysis/test_data_evidence.py
~~~

### 实现 Prompt

~~~text
实现 SDWPF 数据事实分析。必须复用 st_mgprompt.data 中的数据读取、对齐、
时间划分和 mask 语义，禁止重新定义另一套 split。

所有统计量的拟合、阈值、标准化和图构造只使用训练区间。

分析内容：

A. 数据概况
1. 输出训练、验证、测试时间范围、时间点数、窗口数、有效目标比例；
2. 输出 134 台风机 Patv_raw、Wspd 和 Patv_clean_for_input 的均值、标准差、
   分位数、有效率和一阶差分波动率；
3. 输出 data_overview.csv 和 turbine_summary.csv。

B. 时间尺度证据
1. 对每台风机计算 Patv_raw 与 Wspd 的一阶差分；
2. 使用严格因果窗口 6、18、36 计算滚动均值；
3. 计算原序列、趋势和局部残差的高频能量占比、ACF 衰减长度、
   一阶差分标准差；
4. 计算这些表示与未来 H3/H6/H10 目标的相关性或互信息；
5. 不把窗口尾部之后的数据用于当前时刻特征。

C. 同轴对齐证据
1. 构造当前因果同轴 Coarse 表示；
2. 构造仅用于分析的 downsample-then-upsample 表示；
3. 在训练集爬坡事件附近计算峰值时间偏差、局部相关性和重构误差；
4. 输出典型事件曲线，但典型事件必须按预先定义的最大训练集爬坡幅度选择，
   不得人工挑图。

D. 双空间语义证据
1. 复用 graph_prior.py 构造正式 Micro、Macro 和 distance 图；
2. 输出逐节点邻居 Jaccard、全图支持集 Jaccard、边长分布；
3. 计算 Micro 图边上的 Delta-Wspd 余弦相似度和 MA36-Patv 相关性；
4. 计算 Macro 图边上的上述两类语义；
5. 输出 matched 与 crossed semantic-fit 分数；
6. 只描述统计匹配，不描述物理因果。

E. 难度异质性
1. 优先读取可审计的 MovingAverage 或 LightTS 正式预测；
2. 若没有可用预测，则通过已验证 exporter 重新 inference-only 导出；
3. 计算逐视界、逐未来步、逐风机的 Score、MAE、RMSE；
4. 输出 H3/H6/H10 难度差异、134 风机误差分布、Top10%困难风机名单；
5. Top10%阈值由参考模型训练/验证侧规则确定，最终固定后用于测试分析。

输出至少包括：
data_overview.csv、turbine_summary.csv、temporal_statistics.csv、
horizon_feature_association.csv、alignment_event_metrics.csv、
graph_semantic_fit.csv、graph_neighbor_overlap.csv、
baseline_horizon_difficulty.csv、baseline_turbine_difficulty.csv、
analysis_metadata.json。

绘制：
1. 数据划分和分布；
2. 原序列、Fine 局部变化、Coarse 6/18/36 趋势同轴曲线；
3. 频谱和 ACF；
4. 同轴与重采样表示的事件位置比较；
5. Micro/Macro 图与语义匹配；
6. 视界和风机难度分布。

脚本必须支持 --smoke，仅选少量时间和风机验证形状及无泄漏；
正式模式使用完整训练集。所有 CSV、JSON 和图写入 data_evidence 目录。
~~~

### 验收标准

- 所有训练统计不读取验证和测试目标。
- 窗口6、18、36与正式模型定义一致。
- 输出能够直接回答 Fixed Dual、双图和动态损失的设计动机。

---

## 步骤 3 Fixed Dual 与同时间轴设计实验

### 变体定义

| ID | 定义 |
| --- | --- |
| T0 | Canonical Fixed Dual reference |
| T1 | Fine-only |
| T2 | Coarse-only |
| T3 | Shared temporal transform dual |
| T4 | Downsample-upsample Coarse dual |
| T5 | Fixed Dual + direct concat/MLP，无定向交互 |

### 实现 Prompt

~~~text
实现 T0-T5 时间结构实验。

先检查 config.py 和现有 vadsp_gate_mode。Fine-only、Coarse-only、fixed_dual
若已有可靠实现必须复用。只为缺失能力新增最小配置字段。

T0 引用 Canonical，不重新训练。
T1 仅保留 Fine 因果局部分支。
T2 仅保留 Coarse 6/18/36 因果趋势分支。
T3 保留两条分支和参数预算，但让两条分支使用相同时间变换，
   用于排除“仅增加参数”解释。
T4 明确实现非同轴对照：Coarse 先按固定步长聚合，再恢复到 144 位置；
   恢复过程不得读取未来位置，必须记录每个恢复位置来自哪个历史区间。
T5 保留 Fixed Dual，但关闭 Macro Prompt 和双向 Cross-Fusion，
   改为参数规模尽量接近的 concat+MLP。

所有变体固定：
数据、输入列、目标、mask、split、stride、图、扩散阶数、hidden_dim、
coupling 层数、解码器、损失、训练预算、checkpoint 选择和物理裁剪。

为 T4 编写因果访问测试：
修改任一未来输入或未来目标不得改变当前预测；
每个 Coarse 位置不得引用该位置之后的数据。

运行流程：
1. dry-run 唯一差异审计；
2. synthetic smoke；
3. 小规模真实数据 smoke；
4. 正式多 seed 训练；
5. 输出 H3/H6/H10、逐步、逐风机和困难场景预测。

汇总时额外报告：
1. 普通与爬坡场景；
2. 峰值时间偏差；
3. Fine-only 和 Coarse-only 随 horizon 的相对变化；
4. T4 相对 T0 的事件位置误差；
5. 参数量。

不得预设 T0 必须在每个单项指标最优。结论按多视界整体表现和机制证据解释。
~~~

### 验收标准

- T1–T5除声明差异外协议一致。
- T4通过严格因果测试。
- 能区分双尺度信息、同轴对齐和定向交互三种作用。

---

## 步骤 4 双语义图与先验约束实验

### 变体定义

| ID | 定义 |
| --- | --- |
| G0 | Matched Micro/Fine + Macro/Coarse |
| G1 | 两分支共享 Micro 图 |
| G2 | 两分支共享 Macro 图 |
| G3 | Micro/Macro 互换 |
| G4 | 两分支仅使用 distance 图 |
| G5 | 仅使用统计图，不融合 distance |
| G6 | 自由自适应图，不受先验支持集约束 |
| G7 | 固定双图，不做自适应边权重 |
| G8 | 随机重连但保持每行边数 |

### 实现 Prompt

~~~text
实现 G0-G8 双语义空间图实验。

复用 graph_prior.py 和 graph_layers.py。新增字段时优先采用：
branch_graph_assignment = matched/shared_micro/shared_macro/swapped
graph_prior_component = fused/distance_only/statistics_only
adaptive_support_mode = prior_constrained/free/fixed
graph_rewire_mode = none/degree_preserving_random

字段名可根据现有代码风格调整，但必须写入 config、active_config、
protocol report 和 model_summary。

G0 引用 A0。
G1/G2/G3 只改变分支与图的对应关系。
G4 只保留正式距离核及正式 TopK 规则。
G5 只保留训练区间统计相似性。
G6 允许所有非对角节点成为候选，但其他模型容量保持一致。
G7 对应固定先验，不学习边权重。
G8 在每行保持相同非零边数，随机重连；随机图种子必须记录，
   且随机图只作为结构对照，不替换主模型图。

新增测试：
1. G0 的图与现有 Canonical 完全一致；
2. G3 确实交换输入图，而不是只改标签；
3. G4/G5 图来源唯一；
4. G6 不受先验 support mask 限制；
5. G7 无可学习自适应图参数；
6. G8 每行边数与参考图一致且不含未声明自环；
7. 所有图只用训练区间和坐标。

训练并输出：
H3/H6/H10 全指标、逐风机结果、图支持集、最终边权、参数量。

分析：
1. matched、shared、swapped 的性能差异；
2. 数据层 semantic-fit 与模型性能是否方向一致；
3. 高空间依赖与低空间依赖风机组的增益；
4. prior-constrained 与 free adaptive 的边距离、稀疏度和稳定性；
5. random rewiring 用于检验有效空间结构是否优于同稀疏度随机结构。

文字只使用“支持语义匹配设计”“统计关联”“先验约束”，
不声称已经证明尾流传播或物理因果。
~~~

---

## 步骤 5 扩散方向与阶数实验

### 变体定义

| ID | 定义 |
| --- | --- |
| D0 | 无扩散 |
| D1 | 正向一阶 |
| D2 | 转置方向一阶 |
| D3 | 双向一阶 |
| D4 | 双向二阶，即Canonical |
| D5 | 双向三阶 |

### 实现 Prompt

~~~text
实现扩散方向和阶数受控实验 D0-D5。

复用 graph_layers.py 的传播实现，不创建独立图卷积框架。
正向和转置方向仅描述矩阵传播方向，不解释为实际上游或下游风向。

确保每个变体：
1. 使用同一 matched Micro/Macro 先验；
2. 使用同一自适应边权；
3. 仅改变传播方向和阶数；
4. hidden_dim、后续投影维度和其他训练协议一致。

若拼接状态数导致参数量变化，必须同时提供：
1. 原生结构结果；
2. 统一输出投影维度；
3. 参数量报告。
不要求强行使所有参数完全相同，但必须透明报告。

新增测试：
1. 一阶和二阶递推符合定义；
2. 反向路径使用邻接转置；
3. D0没有扩散参数和扩散状态；
4. D5不会错误复用二阶状态；
5. 所有输出形状一致。

输出：
H3/H6/H10、逐步误差、逐风机误差、参数量、峰值显存、
高空间一致场景和局部扰动场景结果。

分析重点：
一阶到二阶是否获得额外收益；
三阶是否发生过度混合；
双向是否优于任一单方向；
扩散收益是否主要出现在空间一致变化场景。
~~~

---

## 步骤 6 Macro Prompt 与 Cross-Fusion 实验

### 变体定义

| ID | 定义 |
| --- | --- |
| F0 | 双分支独立，无共享 |
| F1 | element-wise add |
| F2 | concat + MLP |
| F3 | unified gated fusion |
| F4 | Macro→Fine only |
| F5 | Fine→Coarse only |
| F6 | 双向Cross-Fusion，共享投影 |
| F7 | 双向Cross-Fusion，独立投影，即Canonical |
| F8 | Mean-pooling Macro Prompt |

### 实现 Prompt

~~~text
实现 F0-F8 跨粒度共享方式实验，复用 cross_fusion.py、
coupling_block.py 和现有 A4-A7 实现。

优先直接引用已有正式结果：
F6 对应现有 A7；
F8 对应现有 A4；
涉及近期窗口的补充分析引用 A5、A6。
只有不存在的共享方式才新增训练变体。

所有变体保持 Fine、Coarse、图、扩散、解码、损失和训练协议一致。
F1-F3 应控制融合输出维度为 hidden_dim。
F4 只允许 Fine 从 Macro Prompt 读取。
F5 只允许 Coarse 从最近 Fine 历史读取。
F6 两个方向共用投影。
F7 两个方向使用独立投影。

补充敏感性实验：
cross_fusion_recent_len = 6/12/24/36/48；
macro_prompt_len = 1/2/4/8。
敏感性实验可先用单 seed 筛选，再对关键配置做多 seed。

诊断输出：
1. Macro temporal pooling 权重；
2. 两方向 attention entropy；
3. fusion gate mean/std 和分位数；
4. 两方向交互增量范数；
5. 四个 Macro token 两两余弦相似度；
6. 普通、爬坡、高波动场景分别统计上述诊断。

禁止将历史 Cross-Fusion 注意力直接解释为某一未来 horizon 的注意力。
Horizon 解释必须留给 Node-Horizon Prompt 实验。
~~~

---

## 步骤 7 Node-Horizon Prompt 与解码器实验

### 变体定义

| ID | 定义 |
| --- | --- |
| N0 | 无节点身份 |
| N1 | 无视界身份 |
| N2 | 无节点和视界Prompt |
| N3 | 所有未来步共享同一视界嵌入 |
| N4 | 独立 horizon output heads |
| N5 | last-state decoder，即Canonical |
| N6 | mean-pooling history |
| N7 | attention-pooling history |
| N8 | full-history cross-attention |

### 实现 Prompt

~~~text
实现 Node-Horizon Prompt 和解码器实验。

复用 decoder.py、prompt_alignment.py 和 P0-P5 已有结果。
N5 引用 P0/A0，N8优先引用 P3；容量敏感性引用 P4/P5，
不要重新训练可合法复用的正式结果。

新增配置能力时，确保：
1. node identity 与 horizon identity 可独立关闭；
2. 固定 type embedding 不能被误写成 Fine/Coarse 类型；
3. shared horizon embedding 对所有10个未来步完全相同；
4. independent heads仍为直接多输出，不引入递归或 teacher forcing；
5. mean/attention pooling只读取144步历史；
6.所有解码器输出 [B,10,N]。

输出并分析：
1. 去掉节点Prompt后，Top10%困难风机与其余风机的退化；
2. 去掉Horizon Prompt后，各未来步和H3/H6/H10的退化；
3. last-state、pooling和full-history的精度与参数量；
4. Full-History是否在两种图算子下均同方向退化；
5. horizon embedding 的距离矩阵、范数和预测敏感性；
6. node embedding 与风机难度、平均功率、波动率之间只做相关分析，
   不做物理因果解释。

实现一个 inference-only prompt intervention：
固定同一历史输入，将 horizon embedding 置零或交换，
记录预测变化。该实验只作为模型内部敏感性分析，不称为现实反事实。
~~~

---

## 步骤 8 MS-MG-DWU 分解与替代损失实验

### 变体定义

| ID | 定义 |
| --- | --- |
| L0 | Equal Smooth L1 |
| L1 | 仅视界 difficulty-rate 权重 |
| L2 | 仅风机动态难度权重 |
| L3 | 完整 MS-MG-DWU |
| L4 | 静态视界递增权重 |
| L5 | uncertainty weighting |
| L6 | dynamic weight average |
| L7 | score-aligned hybrid，即现有 A8 对照 |

### 实现 Prompt

~~~text
实现 MS-MG-DWU 的受控分解实验。

当前 A8 同时切换到 masked_score_aligned_hybrid，不能单独隔离视界权重和站点权重。
因此 L0-L3 必须保持相同 Smooth L1 基础损失，只改变两个权重开关。

在 losses.py 中以最小改动支持：
use_horizon_difficulty_weighting
use_node_difficulty_weighting
如果现有实现可通过 granularity_weight_mode 和 site_weight_mode 组合完成，
则不要新增重复字段。

L0：H3/H6/H10等权，所有风机等权。
L1：只启用视界 difficulty-rate。
L2：只启用动态风机权重。
L3：完整 MS-MG-DWU，引用 Canonical。
L4：预先冻结的静态视界权重，权重来源必须在训练前声明。
L5：可学习 uncertainty weighting。
L6：根据相邻 epoch 或更新阶段损失下降率实现 DWA。
L7：引用 A8，不把它解释成单一权重开关。

所有方法使用相同基础预测模型、数据和训练预算。

必须输出：
1. 每个 epoch 的 H3/H6/H10基础损失；
2. 每个 epoch 的视界权重；
3. 风机权重最终值和关键epoch快照；
4. 权重裁剪前后分布；
5. Top10%困难风机和其余风机指标；
6. 全体风机误差分布；
7. H3/H6/H10主指标；
8. 权重是否出现饱和或塌缩。

测试：
1. L0所有权重恒为1；
2. L1风机权重恒为1；
3. L2视界权重恒为1；
4. stop-gradient语义保持；
5. 全无效mask批次不产生NaN；
6. checkpoint恢复包含loss状态和EMA状态。

损失对照必须按变体与视界逐对报告，不得只报告总平均值。
~~~

---

## 步骤 9 困难场景与稳健性分析

### 场景

1. Normal。
2. High-volatility。
3. Ramp-up。
4. Ramp-down。
5. Spatially coherent change。
6. Localized disturbance。
7. 输入随机缺失。
8. 输入连续块缺失。
9. 风速通道轻度噪声。

### 实现 Prompt

~~~text
实现测试集困难场景和稳健性分析，优先基于正式预测导出进行 inference-only 分析。

所有场景定义和阈值只用训练区间拟合，然后冻结到测试集。

High-volatility：
计算训练区间风场平均功率一阶差分绝对值，
主定义使用 mean + 1.5*std；同时输出训练分位数位置供审计。

Ramp：
分别识别 ramp-up 和 ramp-down。
阈值在训练区间确定，事件窗口包含事件点及其相邻预测范围。
输出事件数量、持续长度和覆盖样本数。

Spatially coherent change：
根据同一时刻跨风机 Delta-Patv 方向一致比例与平均幅值定义。

Localized disturbance：
变化集中在少数风机且风场整体变化较小。

对每个场景和每个H3/H6/H10计算：
Score、MAE、RMSE、R²、样本数、有效点数、相对Normal退化率。

Ramp额外计算：
1. 峰值功率误差；
2. ramp幅值误差；
3. ramp发生时间偏差；
4. 事件窗口MAE；
5. ramp-up和ramp-down分开统计。

对输入扰动：
1. 随机缺失比例5%、10%、20%；
2. 连续块长度6、18、36；
3. 仅在测试输入上施加；
4. 不修改目标和mask；
5. 使用预先声明的填充方法；
6. 固定扰动随机种子并记录；
7. 标记为压力测试，不冒充真实故障分布。

至少比较：
A0、A8、Fine-only、Coarse-only、No-Spatial、No-Diffusion、
No/Concat Cross-Fusion、L0、L3，以及正文最强轻量、最强时间序列和最强图基准。
“最强”只能由已冻结验证集规则确定，并在 `scenario_definitions.json` 中写出选择来源；
不得用测试集先选赢家再回填。

除 Normal 外，必须同时保留以下两套难度分组以便复核：
1. 训练集拟合的 Low/Medium/High volatility 阈值；
2. 测试集等频分组（只作描述，不替代训练阈值）；
3. 由同一参考模型和同一 Top10% 规则生成的 Shared difficult Top10% 风机集合。
每组报告 count、share、Score、MAE、RMSE、H3/H6/H10，并把公共测试窗口索引写入
membership 文件，确保所有模型比较同一集合。

复用 evaluate_by_group，但新增通用 group evaluator 和 ramp evaluator。
输出 scenario_definitions.json、scenario_membership.csv、
scenario_metrics.csv、ramp_event_metrics.csv、robustness_metrics.csv。
~~~

---

## 步骤 10 模型内部可解释性分析

### 目标

将解释分为数据解释、结构解释和预测敏感性解释，避免把单一注意力图当成全部可解释性。

### 实现 Prompt

~~~text
扩展 diagnostics.py，生成与模型结构一一对应的解释性诊断。

A. 图解释
1. 保存 Micro/Macro prior 和 final adjacency；
2. 保存支持集Jaccard、权重相关、边距离和每节点TopK；
3. 保存自适应前后边权变化最大的边；
4. 保存 matched/shared/swapped 变体的图语义匹配摘要。

B. Macro Prompt
1. 保存每个样本、节点、历史位置的 pooling 权重摘要；
2. 不必保存全部巨大张量，可保存预先定义场景的代表样本和聚合统计；
3. 输出普通、高波动、ramp-up、ramp-down四类的平均历史权重；
4. 输出4个token的两两余弦相似度和有效秩。

C. Cross-Fusion
1. 保存 Macro→Fine 和 Fine→Coarse attention entropy；
2. 保存门控均值、标准差、分位数；
3. 保存交互增量与原表示的范数比；
4. 按场景、节点难度分组统计。

D. Node-Horizon Prompt
1. 保存节点和未来步嵌入的范数及距离矩阵；
2. 通过置零、交换和均值替换做 inference-only 敏感性；
3. 输出每个horizon的预测变化，不将其称为现实因果效应。

E. MS-MG-DWU
1. 绘制H3/H6/H10权重随epoch变化；
2. 绘制困难和容易风机权重分布；
3. 计算最终权重与独立参考模型风机难度的Spearman相关；
4. 同时报告高权重风机是否真的获得更大误差改善。

所有代表样本选择规则必须预先确定：
普通样本取Normal组中位波动率附近；
高波动样本取固定最高分位；
ramp样本取幅值最大但目标有效的事件；
不得为了图形好看手工挑选。

输出 explanation_manifest.json，记录每张图和每个CSV对应的run、seed、horizon和选择规则。
~~~

---

## 步骤 11 多随机种子、配对 bootstrap 与 DM 检验

### 正式种子

~~~text
[2022, 2023, 2024, 2025, 2026]
~~~

### 实现 Prompt

~~~text
为关键模型运行5个随机种子，并实现配对统计检验。

先保证全部进入正文或核心机制比较的模型至少有 seed=2026；只有在单 seed
矩阵通过静态、协议和结果审计后，才对核心比较补齐 `[2022, 2023, 2024, 2025, 2026]`。
不得因某个 seed 退化而从汇总中删除该 seed。

关键模型至少包括：
A0/T0、最强正文基准、T1、T2、T4、G3、G6、G7、
D0、D3、D4、F2、F6、F7、N0、N1、N8、L0、L1、L2、L3。

资源不足时可先对全部候选运行3个种子，但进入论文核心结论的比较必须补到5个。

多seed汇总：
每个模型、horizon、metric输出mean、std、min、max、seed_count和单seed值。

配对分块bootstrap：
1. 使用相同测试prediction_start_index对齐两个模型；
2. 以完整自然日为重采样块，10分钟数据每块原则上144个时间位置；
3. 保留块内时间顺序和全部风机；
4. 每次重采样重新计算配对指标差；
5. 输出95%置信区间和改善概率；
6. 固定bootstrap随机种子并记录；
7. 不把缺失或无效目标当0。

DM检验：
1. 每个预测起点先在有效风机和前h步内聚合绝对误差或平方误差；
2. 对H3/H6/H10分别构造loss differential；
3. 使用HAC/Newey-West方差，lag至少覆盖预测窗口重叠；
4. 报告统计量、原始p值和Holm校正p值；
5. DM主要用于A0与8个正文基准；
6. 正值方向必须在输出说明中明确定义。

统计输出：
seed_summary.csv、paired_bootstrap.csv、dm_tests.csv、
multiple_comparison_adjustment.csv、statistics_report.md。

结论规则：
同时参考跨seed方向、效应大小和置信区间；
不要只根据单个p值写结论；
若不同seed方向不一致，必须如实报告。
~~~

---

## 步骤 12 复杂度与效率统一审计

### 实现 Prompt

~~~text
在同一硬件、相同batch、相同精度模式下重新进行效率测量。

不得使用P4/P5历史冲突状态中的耗时字段形成正式效率排名。

实现 scripts/empirical_analysis/benchmark_efficiency.py：
1. 加载已完成的best checkpoint；
2. 固定测试batch和输入形状；
3. GPU预热至少50次；
4. 正式测量至少200次；
5. 每次同步CUDA；
6. 报告median、P95和均值延迟；
7. 报告峰值显存；
8. 报告参数量；
9. 报告可训练参数量、模型文件大小和最佳 epoch；
10. 如项目已有可靠MACs/FLOPs工具则复用，否则只报告可验证项目，
   不为追求完整而写不可靠FLOPs；
11. 训练时间只在完整、无中断、相同硬件的run之间比较；P4/P5 等冲突状态只能标记
    `NOT_COMPARABLE`，不能进入正式效率排名。

比较对象：
A0、正文最强轻量基准、T1、T2、D0、D4、F2、F7、N5、N8。

输出 efficiency.csv、hardware_environment.json、efficiency_report.md。
环境报告记录GPU、PyTorch、CUDA、batch、AMP和输入形状，不生成哈希。
~~~

---

## 步骤 13 统一结果表和论文图

### 实现 Prompt

~~~text
从正式CSV/JSON生成论文表格和图片，禁止手工抄写数值。

建立唯一结果加载层：
1. 主指标优先读取metrics_eval_h3/h6/h10.json；
2. 多seed读取seed_summary.csv；
3. 场景结果读取scenario_metrics.csv；
4. 统计检验读取paired_bootstrap.csv和dm_tests.csv；
5. 图诊断读取explanation_manifest.json；
6. 如果预测导出重新计算的数值与正式metrics不一致，
   表格显示正式metrics，样本级图使用导出并标注用途，
   不静默覆盖正式指标。

生成表：
表1 数据和预测协议；
表2 时间尺度与难度数据证据；
表3 总体基准比较；
表4 Fixed Dual；
表5 双语义图；
表6 扩散方向和阶数；
表7 Cross-Fusion共享方式；
表8 Node-Horizon Prompt和解码器；
表9 MS-MG-DWU分解；
表10 困难场景；
表11 多seed和统计检验；
表12 复杂度与效率。

生成图：
1. 数据划分与分布；
2. Fine/Coarse同轴表示；
3. 频谱和ACF；
4. Micro/Macro语义图；
5. 节点与视界难度；
6. Fixed Dual性能；
7. 空间机制；
8. Cross-Fusion机制；
9. Prompt内部诊断；
10. loss权重轨迹；
11. 困难场景；
12. 典型事件预测；
13. 误差分布；
14. 精度-参数量-延迟折中。

视觉规范：
中文论文正文图优先使用一致字体和配色；
H3/H6/H10全篇颜色固定；
误差指标明确向下箭头，R²明确向上箭头；
不使用会夸大差异的截断轴，若截断必须显式标注；
每张图同时导出PNG 300dpi和SVG；
每张图生成source_data.csv；
图中模型名称与正文正式名称一致。

自动检查：
图和表的模型集合、horizon、metric、数值来源、最优方向和小数位；
禁止把reference重复计数；
禁止用平均Score替代分视界主表。
~~~

---

## 步骤 14 撰写论文实证章节

### 推荐章节结构

~~~text
4 数据特征与模型设计依据
4.1 数据集与预测任务
4.2 时间尺度差异及同时间轴建模依据
4.3 Micro与Macro空间语义差异
4.4 节点与预测视界难度异质性

5 实验设计与实证结果
5.1 统一训练协议、评价指标与统计方法
5.2 总体预测性能与显著性
5.3 Fixed Dual与时间对齐机制
5.4 双语义图、先验约束与扩散机制
5.5 Macro Prompt和双向Cross-Fusion
5.6 Node-Horizon Prompt与解码器
5.7 MS-MG-DWU动态优化
5.8 困难场景与稳健性
5.9 模型可解释性
5.10 复杂度与应用边界

6 结果分析与讨论
6.1 为什么需要同时间轴双粒度
6.2 为什么需要语义匹配的空间关系
6.3 跨粒度交互的作用边界
6.4 节点和视界异质性
6.5 稳定性、局限与证据边界
~~~

### 实现 Prompt

~~~text
根据 empirical_analysis_v1 的正式结果撰写论文第4-6章草稿。

写作前读取：
1. data_evidence全部CSV/JSON；
2. summary全部正式表；
3. statistics_report.md；
4. explanation_manifest.json；
5. 当前手稿相关章节。

写作原则：
1. 每节先提出问题，再给实验设置，再报告数值，再解释机制；
2. 明确区分已完成结果、统计推断和解释性观察；
3. 不写计划值，不填虚构数字；
4. 每个定量结论必须能追溯到表或图；
5. 同时报告H3/H6/H10，不只选有利horizon；
6. 说明主指标和辅助指标方向；
7. 多seed写mean±std；
8. 统计显著性必须引用实际检验结果；
9. 图注意力、图边和Prompt只解释为模型内部关联；
10. 单组件消融不写成严格因果贡献；
11. 外部模型对比写成端到端性能，不写成纯组件因果；
12. 结果不支持原假设时修改论证，不修改结果。

每个机制小节采用以下结构：
问题事实 → 对应设计 → 受控对照 → 结果 → 内部诊断 → 证据边界。

输出：
manuscript_empirical_chapter.md
table_figure_callouts.md
result_claim_traceability.csv

result_claim_traceability.csv包含：
claim_id、正文结论、来源文件、表/图编号、模型比较、horizon、metric、
seed范围、统计检验、限制条件。
~~~

## Codex/人工执行 Prompt 拆分与交接协议

为避免一个 Prompt 同时重构框架、实现大量模型、启动长训和写论文，必须按以下顺序
拆分。每个 Prompt 都要先读取本计划中对应的实现 Prompt，再检查仓库当前状态；完成后
生成独立 HANDOFF，不得自动触发下一阶段的正式长训。

~~~text
Prompt 1：步骤0/步骤1 现有资产审计与 empirical_analysis_v1 runner
Prompt 2：步骤2 数据事实与泄漏审计
Prompt 3：步骤3 T_fixed_dual 唯一差异和因果访问测试
Prompt 4：步骤4-5 G/D 图与扩散受控实验
Prompt 5：步骤6-7 F/N Prompt、Cross-Fusion 和解码诊断
Prompt 6：步骤8 L/MS-MG-DWU 分解
Prompt 7：步骤9-10 场景、稳健性和内部解释
Prompt 8：步骤11 多 seed、paired bootstrap、DM
Prompt 9：步骤12 效率测量
Prompt 10：步骤13-14 表图生成与论文章节
Prompt 11：步骤15 最终一致性审计
~~~

步骤3–8的可直接执行版本另存为桌面文件，作为本计划对应 Prompt 的展开版：

~~~text
C:\Users\12811\Desktop\步骤3prompt.md
C:\Users\12811\Desktop\步骤4prompt.md
C:\Users\12811\Desktop\步骤5prompt.md
C:\Users\12811\Desktop\步骤6prompt.md
C:\Users\12811\Desktop\步骤7prompt.md
C:\Users\12811\Desktop\步骤8prompt.md
~~~

每个阶段的 HANDOFF 至少包含：

~~~text
stage、日期、使用的 profile/source_scope、修改文件、只读引用文件、
运行命令、静态测试、普通 smoke、full-shape smoke、正式训练是否启动、
已完成项、未完成项、失败原因、下一阶段入口和用户需要作出的选择
~~~

任何阶段若发现协议冲突、profile 不明、loss 输入不可迁移或 checkpoint 语义不匹配，
必须 `FAIL_CLOSED` 并生成审计报告；不能通过改名、复制指标或隐式降级继续推进。

---

## 步骤 15 最终一致性与完整性审计

### 实现 Prompt

~~~text
对整个 empirical_analysis_v1 执行最终审计，不重新训练，不修改正式结果。

检查：
1. 所有计划变体是否completed或明确记录未完成原因；
2. 每个核心比较是否使用相同数据、split、stride、mask、裁剪和评价；
3. scaler、图、阈值是否train-only；
4. H3/H6/H10是否来自同一10步输出前缀；
5. A0/P0/Canonical是否只计一次；
6. A6/A7/A8定义是否正确；
7. 所有多seed表的seed_count是否真实；
8. bootstrap和DM是否按prediction_start_index配对；
9. 图表数值是否来自正式源；
10. 文稿中的每个数值能否在claim traceability中找到；
11. 是否出现“注意力等于因果”“图边等于尾流”等越界表述；
12. 是否误用P4/P5冲突耗时；
13. 是否有未标记的失败、partial或candidate结果进入正文；
14. 图片是否有source_data.csv和manifest记录。

运行：
compileall；
st_mgprompt现有核心单元测试；
empirical analysis新增测试；
全部CSV/JSON schema检查；
图表来源检查；
claim traceability完整性检查。

输出：
final_audit_report.md
final_artifact_manifest.csv
missing_items.csv

只有当missing_items.csv不存在正文阻断项时，才把实证分析标记为complete。
不得通过填充虚构值、复制其他seed或忽略失败状态来通过审计。
~~~

## 5 建议的正式运行批次

### 批次 A 无训练分析

~~~text
步骤0 资产审计
步骤2 数据事实
已有正式预测的困难场景分析
已有图快照、Prompt和loss诊断整理
现有P/A/E结果重新汇总
~~~

### 批次 B 核心新增训练

~~~text
T1 T2 T4 T5
G1 G2 G3 G6 G7 G8
D1 D2 D3 D5
F0 F1 F2 F3 F4 F5
N0 N1 N2 N3 N4 N6 N7
L0 L1 L2 L4 L5 L6
~~~

T0、G0、D4、F6/F7/F8、N5/N8、L3/L7优先复用现有合法正式结果。
步骤3–8的正式训练须在静态审计、smoke 和 full-shape 通过后由用户单独批准。

### 批次 C 多种子补充

先完成单seed完整矩阵，确认配置和结果正常后，再对核心比较运行：

~~~text
--seeds 2022 2023 2024 2025 2026
~~~

### 批次 D 统计、图表和论文

必须在预测导出对齐和正式指标审计完成后开始。

每一批结束都必须写入阶段性 HANDOFF，且不把 smoke、candidate、partial 或失败结果
写进正文主表。

## 6 最小可发表集合

如果计算资源无法一次完成全部扩展，以下项目不可省略：

1. 数据层时间尺度、空间语义和难度异质性证据。
2. Fine-only、Coarse-only、Resampled-dual和Canonical。
3. Shared Micro、Shared Macro、Swapped和Matched双图。
4. 无扩散、双向一阶、双向二阶。
5. concat、单向交互、双向独立投影。
6. 无节点Prompt、无Horizon Prompt、last-state与full-history。
7. Equal、horizon-only、node-only和完整MS-MG-DWU。
8. Normal、high-volatility、ramp-up、ramp-down。
9. 五随机种子的核心模型。
10. 配对分块bootstrap和A0对主要比较对象的DM检验。
11. 图、Prompt、门控和loss权重诊断。

## 7 最终完成定义

只有同时满足以下条件，才认为“完整实现实证分析”：

- 数据动机、总体性能、组件机制、困难场景、统计稳定性、内部解释和效率边界均有正式产物。
- 每个模型设计都至少有一项数据证据和一项受控模型实验。
- 所有正文结论都可追溯到正式结果文件。
- 核心比较完成五随机种子。
- 每个正式 run 都满足 §2.2 artifact contract，并能由 `source_scope` 与
  `protocol_profile` 判定可比较范围。
- 正文不存在把相关性写成因果的表述。
- 所有图表均由脚本生成且有source data。
- 最终审计无正文阻断项。
