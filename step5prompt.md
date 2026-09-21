# 步骤5 Prompt：扩散方向与阶数实验

## 0. 使用方式与范围

执行目录为 `D:\PaperProject\GyxPaper2`。本 Prompt 只实现 D0–D5 的传播方向和阶数受控
实验，复用步骤4已经冻结的 matched Micro/Macro 图和自适应边权；不重新构造图，不修改
时间粒度、Prompt、decoder 或 loss。完成后生成 `HANDOFF_STEP5.md`，正式长训须由用户
单独批准。

## 1. 研究问题

验证以下机制，而不是把传播方向赋予地理含义：

1. 图传播是否优于没有显式扩散状态的对照？
2. 邻接矩阵方向与转置方向各自是否提供互补表示？
3. 双向一阶到双向二阶是否有稳定增益，三阶是否出现过度混合？
4. 扩散收益是否主要出现在空间一致变化，而不是所有样本平均提升？

正文只能使用“矩阵传播方向”“一阶/二阶状态”“空间一致场景中的误差变化”等表述，
不能写成上游、下游、风向传播或物理尾流证明。

## 2. 必读文件

~~~text
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/graph_layers.py
custom_models/src/st_mgprompt/graph_prior.py
custom_models/src/st_mgprompt/model.py
custom_models/src/st_mgprompt/coupling_block.py
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/empirical_protocol.py
custom_models/src/st_mgprompt/run_empirical.py
custom_models/src/st_mgprompt/evaluate.py
custom_models/src/st_mgprompt/diagnostics.py
custom_models/tests/empirical_analysis/
~~~

先确认当前 `graph_operator`、`diffusion_order_micro`、`diffusion_order_macro`、
`diffusion_use_bidirectional` 和邻接转置的真实实现。若当前实现只有
`simple/bidirectional_diffusion` 两种模式，新增方向字段和传播函数必须保持向后兼容，
并在旧正式配置上运行回归测试。

## 3. 固定协议和传播定义

所有 D 变体继承 Canonical/T0 的数据、mask、batch、stride、loss、decoder、hidden_dim、
图 prior 和自适应 support。步骤4的 G0 图是唯一图来源。每个 run 写入：

~~~text
protocol_profile
source_scope=internal_diffusion
paired_reference=D4/T0
graph_identity_reference=G0/CANONICAL
operator_direction
diffusion_order_micro
diffusion_order_macro
diffusion_use_bidirectional
~~~

建议把传播定义写成显式递推。设归一化邻接为 `A`、转置为 `Aᵀ`、输入状态为 `H⁽⁰⁾`：

~~~text
forward_state[k]  = A   @ forward_state[k-1]
reverse_state[k]  = Aᵀ  @ reverse_state[k-1]
~~~

D1 只保留 `forward_state[1]`，D2 只保留 `reverse_state[1]`，D3 保留两者的一阶状态，
D4 保留两方向到二阶状态，D5 保留两方向到三阶状态。D0 不产生传播状态；如果仍需要
图输入以保持模型形状，必须明确它是 simple/local graph control，而不是隐藏的 diffusion。

## 4. 变体合同

| 变体 | 定义 | 唯一差异 |
| --- | --- | --- |
| D0 | 无扩散 | 不创建 diffusion state/parameter |
| D1 | 正向一阶 | forward 一阶 |
| D2 | 转置方向一阶 | reverse/transpose 一阶 |
| D3 | 双向一阶 | forward+reverse 一阶 |
| D4 | 双向二阶 | Canonical/reference |
| D5 | 双向三阶 | forward+reverse 到三阶 |

如果拼接状态数造成参数量变化，必须同时保存：原生结构结果、统一输出投影结果、
参数量和显存；不能只挑参数相同的版本而不说明另一个版本。

## 5. 实现任务

### 5.1 统一传播内核

1. 在 `graph_layers.py` 中复用现有传播函数；若需要扩展，使用一个可配置的方向/阶数
   内核，不新建第二套图卷积框架。
2. 明确状态列表长度、拼接顺序、输出投影和 bias 初始化；写入 `model_summary.json`。
3. D0 的 forward 中不得创建或更新 diffusion parameter；D5 必须从一阶状态递推到三阶，
   不能重复使用二阶张量冒充三阶。
4. `A` 与 `Aᵀ` 的归一化规则必须记录；不能对转置图重新做未经声明的 TopK。
5. 所有变体都接收同一个 G0 graph identity 和同一个自适应 edge-weight 语义。

### 5.2 参数预算和容量报告

1. 若所有阶数使用统一输出投影，保存 `projection_mode=shared_output_dim`。
2. 若原生 concat 造成参数/激活增大，保存 `projection_mode=native` 并报告差异。
3. 训练和推理显存、参数量、状态数和输出维度都写入效率 manifest。
4. 不通过减少 hidden_dim、层数、batch 或输入长度来伪造公平。

### 5.3 变体注册

1. 将 D0–D5 添加到 `empirical_protocol.py`，设置 `expected_unique_diff`。
2. 对 D1–D5 使用明确字段，例如 `diffusion_direction` 和 `diffusion_order`；字段必须
   能在 `effective_config` 中还原实际 forward。
3. D4 使用 reference-only 方式引用 Canonical，不能重新训练或复制 checkpoint。
4. 对 D0 的 `model_summary` 写 `diffusion_state_count=0`，对 D1–D5 写真实状态数。

## 6. 必须实现的测试

新增 `custom_models/tests/empirical_analysis/test_step5_diffusion.py`：

1. 给定固定小邻接矩阵，验证一阶和二阶递推数值与手算结果一致。
2. 验证 reverse 使用 `Aᵀ`，不是再次使用 `A`。
3. D0 的传播状态数为0且没有 diffusion 参数梯度。
4. D1/D2/D3 的状态方向集合严格符合定义。
5. D5 的三阶状态确实依赖二阶递推，不能与 D4 完全相同。
6. 所有变体输出 `[B,10,134]` 且有限值；混合精度和 CPU smoke 都通过。
7. G0 graph identity、mask、clip、batch、stride、loss 和 decoder 未改变。
8. 对自适应边权做同一输入回归，确认只改变 diffusion 字段，不改变 support。
9. 传播矩阵不会读取验证/测试期统计或未来目标。

测试必须同时覆盖稠密小图和 134 节点稀疏形状；只测一维 toy tensor 不算通过。

## 7. 运行顺序和命令模板

~~~powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step5_diffusion.py -q
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --dry-run
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --smoke
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --full-shape
~~~

若当前 runner 尚未有 D family，先补齐 registry、CLI 选择和输出路径。用户批准后才执行：

~~~powershell
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D5 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_diffusion `
  --run-full --skip-completed
~~~

## 8. 产物要求

~~~text
custom_models/results/empirical_analysis_v1/D/D_VARIANT_MANIFEST.json
custom_models/results/empirical_analysis_v1/D/D_PROTOCOL_AUDIT.json
custom_models/results/empirical_analysis_v1/D/D_OPERATOR_TRACE.csv
custom_models/results/empirical_analysis_v1/D/D_STATE_COUNT_SUMMARY.csv
custom_models/results/empirical_analysis_v1/D/D_PERFORMANCE_SUMMARY.csv
custom_models/results/empirical_analysis_v1/D/D_SCENARIO_METRICS.csv
custom_models/results/empirical_analysis_v1/D/D_EFFICIENCY_SUMMARY.csv
custom_models/results/empirical_analysis_v1/D/D_FAILURES.csv
custom_models/results/empirical_analysis_v1/D/HANDOFF_STEP5.md
~~~

每个正式 run 还要保存实际图身份、传播方向、状态数量、参数量、峰值显存和
PLAN.md §2.2 的完整 artifact contract。D4 只能写 reference/source manifest，不复制
Canonical checkpoint。

## 9. 分析与验收

必须自动报告 H3/H6/H10 的 Score、MAE、RMSE、R²、逐步误差、逐风机误差、空间一致变化、
局部扰动和 High-volatility 分组。报告：

- D1/D2 相对 D0 的变化；
- D3 相对单方向的变化；
- D4 相对 D3 的额外收益；
- D5 是否出现过度混合、参数/显存代价或性能退化；
- 传播收益是否集中在空间一致场景。

步骤5完成条件：

1. 传播递推和方向测试通过；
2. D0 无隐藏扩散状态，D5 无错误状态复用；
3. 所有变体只改变声明的 direction/order；
4. 正式结果、失败项和效率数据均可追溯；
5. `HANDOFF_STEP5.md` 说明是否启动正式训练及未完成项。

任何将矩阵方向解释成物理方向、未声明的邻接重构、状态数不一致或未来信息进入图/传播，
都必须 FAIL_CLOSED。
