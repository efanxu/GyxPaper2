# 步骤8 Prompt：MS-MG-DWU 分解与替代损失实验

## 0. 使用方式与边界

执行目录固定为 `D:\PaperProject\GyxPaper2`。本 Prompt 只实现 L0–L7 的损失分解、权重
状态记录和替代损失对照；模型结构、图、扩散、Prompt、decoder 和其他章节内容必须
保持独立。完成后生成 `HANDOFF_STEP8.md`，不得自动启动其他章节的长训。

## 1. 研究问题

本步骤要区分 MS-MG-DWU 的两个维度：

1. 不同预测 horizon 的 difficulty/rate weighting 是否有用？
2. 不同风机的 dynamic node weighting 是否有用？
3. 完整 MS-MG-DWU 是否比等权、静态权重、uncertainty weighting 和 DWA 更稳定？
4. 权重是否真实响应误差难度，还是出现饱和、塌缩或把无效 mask 当成困难样本？

MS-MG-DWU 是方法内部的训练优化机制，不得把动态权重解释成真实风机重要性或物理
风险因果。A8 是独立的 `w/o MS-MG-DWU` Batch4 formal artifact，不能被简化成单一
“site off”或“horizon off”开关。

## 2. 必读文件和当前 loss 审计

~~~text
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/losses.py
custom_models/src/st_mgprompt/metrics.py
custom_models/src/st_mgprompt/formal_runner.py
custom_models/src/st_mgprompt/run_empirical.py
custom_models/src/st_mgprompt/empirical_protocol.py
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/a8_batch4_contract.py
custom_models/src/st_mgprompt/a8_batch4_readiness.py
custom_models/src/st_mgprompt/check_protocol.py
custom_models/src/st_mgprompt/diagnostics.py
custom_models/results/st_mgprompt_uniform_bs4/
custom_models/results/st_mgprompt_canonical/
custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md
~~~

先追踪训练器如何创建 loss、如何执行全无效 mask 批次、如何保存/恢复 loss state、
optimizer state、AMP state、EMA 和 early-stopping 状态。确认 `MSMGDWULoss.last_details`
与 `diagnostics_state()` 的实际写入路径；不能只新增 CSV 而不保证 checkpoint 恢复。

## 3. 固定协议和 loss contract

L0–L6 使用相同 Canonical 基础模型、输入、graph、decoder、数据 split、stride、batch、
seed、训练预算和 checkpoint selection。默认：

~~~text
msmg_base_loss = smooth_l1
eval_horizons = [3,6,10]
loss_protocol = method_full
mask = valid_target_mask
use_msmg_dwu = true（L0-L6 仍使用同一个 loss container，只切换权重模式）
lambda_site、ema_alpha、node_weight_clip、granularity_weight_clip 固定
~~~

如果为了实现 L0 等权而创建独立 `masked_smooth_l1` 路径，必须证明它与
`MSMGDWULoss(granularity_weight_mode=static, site_weight_mode=static)` 的基础项相同；
不能让 L0 因 loss container 不同而改变 reduction、mask 或有效样本定义。

每个 run 写入：`loss_identity`、`base_loss`、`granularity_weight_mode`、
`site_weight_mode`、`lambda_site`、EMA/clip 参数、`loss_state_schema_version`、
`protocol_profile` 和 `source_scope=internal_loss`。

## 4. 变体合同

| 变体 | 定义 | 真实开关/要求 |
| --- | --- | --- |
| L0 | Equal Smooth L1 | horizon static + node static，所有有效权重为1 |
| L1 | horizon difficulty-rate only | horizon difficulty/rate，node static |
| L2 | dynamic node difficulty only | horizon static，node dynamic |
| L3 | 完整 MS-MG-DWU | difficulty-rate + dynamic node，引用 Canonical |
| L4 | static increasing horizon weights | 训练前冻结的 horizon 权重，node static |
| L5 | uncertainty weighting | 可学习 `log_sigma_g`，node static |
| L6 | dynamic weight average | 相邻 epoch/更新阶段损失比率，node static 或按合同固定 |
| L7 | score-aligned hybrid | 只读引用 A8，独立 Batch4 scope，不作为单开关因果证据 |

L0–L6 的基础模型必须一致；L7 不能与 L0–L6 放进同一 profile 的组件因果表。

## 5. 实现任务

### 5.1 统一 loss state 和权重接口

1. 在 `losses.py` 中以最小改动扩展 `MSMGDWULoss`，优先复用现有
   `granularity_weight_mode` 和 `site_weight_mode`。
2. 若新增 `static_increasing` 或 `dynamic_weight_average`，同时更新 config 校验、
   loss constructor、`get_loss_fn`、序列化和 diagnostics state；不得仅在 runner 中
   临时计算而绕过 loss。
3. 所有权重只在有效 mask 上更新；全无效 batch 返回 `None`、跳过 backward/optimizer step，
   并记录 `skipped_all_invalid_batches`。
4. stop-gradient 语义必须保持：动态权重用于加权但不通过权重路径反向传播，除非
   uncertainty `log_sigma_g` 明确是可学习参数。
5. 权重裁剪上下界、归一化方式、EMA alpha、温度和 gamma 写入 effective config。

### 5.2 L0–L3 分解

1. L0：horizon weight、node weight 均为1；`last_details` 和诊断文件必须证明恒等。
2. L1：只启用 horizon difficulty/rate；每个 node weight 恒为1。
3. L2：只启用 node dynamic；每个 horizon weight 恒为1。
4. L3：引用 Canonical 的完整 MS-MG-DWU，不能重新创建“近似完整”版本。
5. 每个 epoch 记录 H3/H6/H10 的 raw loss、EMA loss、initial loss、difficulty level、
   relative rate、最终权重和 weighted contribution。

### 5.3 L4 静态 horizon 权重

1. 在训练开始前声明权重来源：等差、预设比例或仅由训练集先验统计计算。
2. 权重一旦写入 `resolved_config.json` 就冻结；验证/测试误差不得改变它。
3. 记录归一化前后权重、来源规则和 horizon 映射；node weight 恒为1。
4. 若现有 config 不支持该模式，新增枚举和单元测试，不把它伪装成 difficulty-rate。

### 5.4 L5 uncertainty weighting

1. 复用 `log_sigma_g` 参数和 checkpoint state；只允许 horizon uncertainty 更新。
2. node weight 恒为1，`lambda_site` 和基础 Smooth L1 保持固定。
3. 记录 `log_sigma_g`、`exp(-log_sigma_g)`、裁剪/归一化状态和每 epoch 变化。
4. 检查 checkpoint reload 后 `log_sigma_g` 与 EMA/optimizer state 完全恢复。

### 5.5 L6 dynamic weight average

1. 明确 DWA 更新时机：epoch 结束或固定更新阶段；禁止每个 batch 隐式改变而不记录。
2. 使用上一阶段/epoch 的有效 raw loss 比率，明确温度和初始权重；无有效 loss 时保持上一状态。
3. 权重只能来自训练区间 loss，不能读取 validation/test 误差。
4. 保存更新前后 loss、比率、softmax 权重、clip 和归一化结果。
5. 若 DWA 与现有 `difficulty_rate` 共享字段会混淆，使用独立枚举和 loss-state schema。

### 5.6 L7/A8 边界

1. 只读审计 A8 dedicated contract：`training_batch_profile_id=uniform_train_batch4_v1`、
   4/4/4 batch、独立 run-id、loss identity 和 readiness。
2. L7 只能写 `paired_reference=A8`，不能声称单独关闭 horizon 或 node weighting。
3. 不复制 A8 checkpoint/metrics，不把 A8 的 Batch4 数值混入 L0–L6 的 32/4/4 表。

## 6. 必须实现的测试

新增 `custom_models/tests/empirical_analysis/test_step8_losses.py`：

1. L0 的所有 horizon/node 权重始终为1，total loss 等于 masked Smooth L1 reduction。
2. L1 的 node weight 恒为1，L2 的 horizon weight 恒为1。
3. L3 的权重更新、EMA、clip、stop-gradient 与 Canonical loss state 一致。
4. L4 权重在训练全过程不变，且不读取 validation/test。
5. L5 的 `log_sigma_g` 可学习、可保存、可恢复，node weight 恒为1。
6. L6 的 DWA 只在声明时机更新，首个有效阶段有明确初始化。
7. 全无效 mask 批次不会产生 NaN，不执行 optimizer step，并记录跳过数量。
8. 断点恢复后 loss、EMA、权重、optimizer、AMP 和随机状态与不中断路径一致。
9. 权重 clip 前后分布、饱和比例和有效 horizon/node 数可审计。
10. 所有变体预测 shape `[B,10,134]`，模型结构和数据协议完全相同。
11. L7/A8 contract audit 通过但不会被误归入 L0–L6。

测试需覆盖 synthetic loss、真实小 batch、CPU/AMP smoke 和 checkpoint reload；只检查最终
metrics 不算通过。

## 7. 运行顺序和命令模板

~~~powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step8_losses.py -q
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --dry-run
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --smoke
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --full-shape
~~~

L3/L7 默认只读引用 Canonical/A8；用户批准后才执行新训练：

~~~powershell
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L4 L5 L6 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_loss `
  --run-full --skip-completed
~~~

本 Prompt 不启动其他章节的跨模型训练；只完成本步骤的损失实现、审计和替代损失对照。

## 8. 产物要求

~~~text
custom_models/results/empirical_analysis_v1/L/L_VARIANT_MANIFEST.json
custom_models/results/empirical_analysis_v1/L/L_PROTOCOL_AUDIT.json
custom_models/results/empirical_analysis_v1/L/L_LOSS_INPUT_CONTRACT.json
custom_models/results/empirical_analysis_v1/L/L_EPOCH_LOSS.csv
custom_models/results/empirical_analysis_v1/L/L_HORIZON_WEIGHTS.csv
custom_models/results/empirical_analysis_v1/L/L_NODE_WEIGHTS.csv
custom_models/results/empirical_analysis_v1/L/L_WEIGHT_DISTRIBUTION.csv
custom_models/results/empirical_analysis_v1/L/L_CHECKPOINT_RESTORE_AUDIT.json
custom_models/results/empirical_analysis_v1/L/L_PERFORMANCE_SUMMARY.csv
custom_models/results/empirical_analysis_v1/L/L_DIFFICULTY_GROUP_METRICS.csv
custom_models/results/empirical_analysis_v1/L/L_FAILURES.csv
custom_models/results/empirical_analysis_v1/L/HANDOFF_STEP8.md
~~~

每个 completed run 还必须有 PLAN.md §2.2 的完整 artifact contract；`run_status`、实际
loss identity、权重 state schema 和 A8 scope 必须能从 manifest 还原。

## 9. 结果分析和验收

自动报告 H3/H6/H10 的 Score、MAE、RMSE、R²、raw/base loss、weighted contribution、
Top10% 困难风机与其余风机指标、权重饱和/塌缩和权重与独立参考难度的 Spearman 相关。
逐模型逐 horizon 报告，不只给总平均。

步骤8完成条件：

1. L0–L6 只有声明的权重机制差异，基础 Smooth L1、模型和数据协议一致；
2. 每个 epoch 的 loss/weight/state 可追溯，checkpoint reload 测试通过；
3. L7/A8 被标为独立 Batch4 scope，不冒充单开关消融；
4. 全无效 mask、clip、stop-gradient 和失败状态均有记录；
5. `HANDOFF_STEP8.md` 列出实际结果、未完成项和后续步骤入口。

任意 loss 读取未来/测试难度、权重未恢复、L0 权重不为1、或把 A8 写成 L2/L1 单组件，
必须 FAIL_CLOSED。
