# 步骤7 Prompt：Node-Horizon Prompt 与解码器实验

## 0. 使用方式与边界

执行目录为 `D:\PaperProject\GyxPaper2`。本 Prompt 只实现 N0–N8 的节点/视界 Prompt 和
历史解码器对照；不重新实现图、扩散、Macro Prompt、Cross-Fusion 或 MS-MG-DWU。完成后
生成 `HANDOFF_STEP7.md`，inference-only intervention 不得被称作现实反事实。

## 1. 研究问题

本步骤回答：

1. 节点身份是否帮助模型区分 134 台风机的固定差异？
2. 不同未来步是否需要不同的 horizon identity？
3. last-state、mean/attention pooling 和 full-history cross-attention 的精度/容量折中是什么？
4. Prompt 改动对预测的敏感性是否能通过受控置零/交换实验观察？

节点或 horizon embedding 与误差、平均功率、波动率之间只做相关分析；不能声称 embedding
具有物理含义，也不能把 intervention 解释为现实世界因果效应。

## 2. 必读文件和当前实现核对

~~~text
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/prompt_alignment.py
custom_models/src/st_mgprompt/decoder.py
custom_models/src/st_mgprompt/model.py
custom_models/src/st_mgprompt/coupling_block.py
custom_models/src/st_mgprompt/check_protocol.py
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/empirical_protocol.py
custom_models/src/st_mgprompt/run_empirical.py
custom_models/src/st_mgprompt/diagnostics.py
custom_models/results/st_mgprompt_precision/
custom_models/results/st_mgprompt_component_ablation/
custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md
~~~

必须追踪 `STPromptEmbedding`、`STPromptDirectDecoder`、`HorizonDirectDecoder`、
`STPromptFullHistoryDecoder` 的真实 shape 和 metadata。确认当前配置能否独立关闭 node
identity、horizon identity、granularity/type embedding；如果不能，新增显式字段，而不要
复用一个含义不清的 `horizon_only` 兼容模式。

## 3. 固定协议和 reference 规则

所有新训练变体继承 Canonical 的输入、图、扩散、Cross-Fusion、loss、split、stride、
batch、seed、clip 和 checkpoint selection；`protocol_profile=STMG_FORMAL_V2`、
`source_scope=internal_decoder`。

正式结果优先只读引用：

~~~text
N5 = P0/A0 = last-state + full ST Prompt（reference-only）
N8 = P3 = full-history cross-attention（若 semantic audit 通过，reference-only）
P4/P5 = full-history capacity references，只用于容量/效率上下文
~~~

任何旧目录若存在 Single-token Macro Prompt、Horizon-only Prompt 或旧 decoder 语义，
必须标记 `semantic_mismatch`，不得按名称复用。

## 4. 变体合同

| 变体 | 定义 | 实现要求 |
| --- | --- | --- |
| N0 | 无节点身份 | full mode，关闭 node embedding，保留 horizon/type 语义 |
| N1 | 无视界身份 | 每个未来步使用同一 horizon embedding；不改变 node identity |
| N2 | 无节点和视界 Prompt | 关闭 node/horizon identity，保留必要的 type/shape 适配或显式 zero prompt |
| N3 | 共享 horizon embedding | 10 个未来步共享同一可学习 horizon 向量 |
| N4 | 独立 horizon output heads | `direct_multi_output_horizon_head`，直接输出10步，无递归 |
| N5 | last-state + ST Prompt | Canonical/reference |
| N6 | mean-pooling history | 只读144步历史后做 mean pooling |
| N7 | attention-pooling history | 只读144步历史后做 attention pooling |
| N8 | full-history cross-attention | 只读144步历史做 per-turbine cross-attention |

N1 与 N3 的区别必须写清：N1 是“去掉步特异身份、使用共享 horizon 表示”的消融，
N3 是专门控制共享 horizon 向量的实现对照；若两者代码完全相同，必须合并命名或解释
为什么仍保留两种 protocol，不能让读者误以为是不同机制。

## 5. 实现任务

### 5.1 统一 Prompt contract

1. 明确 Prompt 的 shape 为 `[1,H,N,D]`，预测输出为 `[B,H,N]`；所有 horizon 都由同一
   10 步输出的前缀评估。
2. 为 node identity、horizon identity 和 granularity/type embedding 设置独立的配置字段，
   并在 `effective_config`、decoder metadata 和 protocol check 中写出。
3. 固定 `type/granularity embedding` 不能被误写成 Fine/Coarse 的动态分支标识；其语义
   只能是已声明的输入类型编码。
4. 任何 N0–N3 都不得改变历史窗口、图、融合、hidden_dim 或输出 heads 以外的字段。

### 5.2 N0–N3 Prompt 消融

1. N0 关闭 node embedding 后，prompt 在节点维度共享，但 horizon/type 部分仍按合同保留。
2. N1/N3 将 10 个未来步的 prompt 替换为同一向量；保存每个未来步的 embedding 差异，
   确认差异确实为零（训练 dropout 关闭后验证）。
3. N2 必须同时关闭 node 和 horizon identity；如果保留一个全局可学习 prompt，需命名为
   `shared_non_identity_prompt`，不能称为“无 Prompt”。
4. 输出 node/horizon embedding 范数、距离矩阵、跨节点/跨 horizon 方差和有效秩。

### 5.3 N4 独立 horizon heads

1. 复用 `HorizonDirectDecoder` 或最小扩展，使用一个直接多输出 head 产生 10 步。
2. 禁止 teacher forcing、autoregressive loop、未来观测输入和递归反馈。
3. 记录每个 head 的参数量、输出 shape、初始化和共享层；不能把共享线性层重复复制而
   误称为 independent heads。

### 5.4 N5–N8 历史解码器

1. N5 只读引用 Canonical last-state decoder。
2. N6 对 Fine/Coarse 历史沿时间维做 mean pooling；N7 使用 attention pooling，保存
   attention weight 的历史索引范围和 entropy。
3. N8 只使用 144 步历史的 full-history cross-attention；复用 `STPromptFullHistoryDecoder`
   和 P3 正式结果，若需新训练必须保留 full-history 的容量说明。
4. 所有 decoder 的 `teacher_forcing=False`、`autoregressive=False`、
   `future_observed_features_used=False` 必须由 `check_protocol.py` 验证。
5. decoder 输出统一为 `[B,10,N]`，不能在某个变体改为 `[B,N,10]` 而不经过统一 adapter。

### 5.5 Inference-only intervention

实现独立脚本 `scripts/empirical_analysis/prompt_intervention.py`：

1. 固定同一批历史输入、模型权重、随机状态和 eval 模式。
2. 对 horizon prompt 做 zero、循环交换、均值替换；对 node prompt 做 zero、节点交换。
3. 记录每次 intervention 的预测差、MAE/Score 变化、按 horizon 和节点汇总。
4. 不修改 checkpoint，不反向传播，不改变 target/mask，不把 intervention 后预测写回正式指标。
5. 输出 `intervention_type`、来源 run、seed、样本选择规则和“内部敏感性分析”标签。

## 6. 必须实现的测试

新增 `custom_models/tests/empirical_analysis/test_step7_prompt_decoder.py`：

1. N0 的节点维度 prompt 在 eval 模式下完全共享，N1/N3 的 horizon prompt 在 10 步间共享。
2. N2 同时没有 node/horizon identity；type/granularity 字段语义仍可审计。
3. N4 输出 `[B,10,N]`，无 teacher forcing、无 autoregressive 和未来观测输入。
4. N5/N6/N7 只读取历史最后状态或 144 步历史；N8 cross-attention 的 key/value 长度正确。
5. N8 的 attention 权重可追溯到历史位置，不能访问预测目标或未来输入。
6. 所有 decoder 输出有限值，H3/H6/H10 来自同一 10 步 prediction 前缀。
7. zero/swap intervention 在固定输入下可重复，诊断脚本不会改变原预测。
8. 唯一差异审计不允许改变 graph、fusion、loss、batch、stride、clip 或 checkpoint 规则。

## 7. 运行顺序和命令模板

~~~powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step7_prompt_decoder.py -q
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --dry-run
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --smoke
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --full-shape
~~~

正式训练只对未能合法复用且 smoke 通过的 N0–N4/N6/N7 执行：

~~~powershell
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N6 N7 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_decoder `
  --run-full --skip-completed
python scripts/empirical_analysis/prompt_intervention.py `
  --run <completed_run_dir> --mode zero,swap,mean --split test
~~~

N5/N8/P4/P5 的正式引用由语义审计和 evaluate-only 导出完成；不因缺少诊断而重训。

## 8. 产物要求

~~~text
custom_models/results/empirical_analysis_v1/N/N_VARIANT_MANIFEST.json
custom_models/results/empirical_analysis_v1/N/N_PROTOCOL_AUDIT.json
custom_models/results/empirical_analysis_v1/N/N_PROMPT_EMBEDDING_SUMMARY.csv
custom_models/results/empirical_analysis_v1/N/N_HORIZON_DISTANCE.csv
custom_models/results/empirical_analysis_v1/N/N_NODE_DIFFICULTY_CORRELATION.csv
custom_models/results/empirical_analysis_v1/N/N_DECODER_SUMMARY.csv
custom_models/results/empirical_analysis_v1/N/N_INTERVENTION_SUMMARY.csv
custom_models/results/empirical_analysis_v1/N/N_PERFORMANCE_SUMMARY.csv
custom_models/results/empirical_analysis_v1/N/N_FAILURES.csv
custom_models/results/empirical_analysis_v1/N/HANDOFF_STEP7.md
~~~

每个 formal/evaluate-only run 还必须满足 PLAN.md §2.2 artifact contract；诊断 manifest
记录 run、seed、horizon、节点抽样、样本选择规则和 intervention 类型。

## 9. 结果分析和验收

自动报告：

- N0 相对 N5 在 Top10% 困难风机与其他风机的退化；
- N1/N3 相对 N5 的逐未来步和 H3/H6/H10 变化；
- N4 与 N5 的精度、参数量和 head-specific error；
- N6/N7/N8 与 last-state 的历史使用方式、性能和显存差异；
- embedding 距离/范数/有效秩与风机难度、平均功率、波动率的相关性；
- intervention 的预测敏感性，明确标为内部分析。

步骤7完成条件：

1. node/horizon/type embedding 三者语义和开关独立可审计；
2. 所有 decoder 无未来观测、无 teacher forcing、输出 shape 统一；
3. N8 的 full-history 只读 144 步历史，N5/N8 reference 语义经过审计；
4. intervention 不改变正式预测和指标；
5. 完整 artifact、失败记录和 `HANDOFF_STEP7.md` 已生成。

任意 horizon identity 未真正关闭、解码器访问未来、intervention 覆盖正式指标或把 embedding
相关性写成物理含义时，必须 FAIL_CLOSED。
