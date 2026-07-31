# ST-MGPrompt Fixed Dual 正式实验体系重构交接

更新时间：2026-07-25（Asia/Shanghai）

## 1. 当前结论

代码重构、Canonical 迁移、A0/P0 引用、配置矩阵、普通 smoke、指令要求的 full-shape preflight、静态审计和正式运行说明均已完成。

唯一完整模型：

```text
FULL_FIXED_DUAL_DYNAMIC_MSMGDWU
```

唯一 canonical run-id：

```text
full_fixed_dual_keep_msmgdwu_seed2026
```

关系：

```text
旧 A1 artifact
    -> 唯一 CANONICAL_FULL
    -> A0 reference
    -> P0 reference

新 P1-P5
    -> Fixed Dual 下重新训练（尚未启动正式长训）

新 A1-A8
    -> Fixed Dual 下重新训练（尚未启动正式长训）
```

正式目录当前只有 reference、effective config、diff、正式命令和 dry-run 状态；P1-P5、A1-A8 的正式 checkpoint/metrics 数量为 0。

## 2. 指令恢复与环境覆盖

桌面文件：

```text
C:\Users\12811\Desktop\codex任务指令.txt
```

实际为 0 字节。完整原始指令已从交接文件记录的附件路径恢复：

```text
C:\Users\12811\.codex\attachments\a638d24c-c48b-4a5a-a983-5fdf22668239\pasted-text.txt
```

原协议解释器：

```text
D:\Apps\Miniconda3\envs\env_tslib\python.exe
```

当前 Windows 推荐解释器为：

```text
D:\Apps\Miniconda3\envs\env_tslib\python.exe
```

所有审计、smoke 和正式 runner 均由该解释器启动；Python 子进程继承父进程的 `Path(sys.executable).resolve()`，不读取预生成命令中的解释器字段，也不跨环境借用 base `site-packages`。

实际验证环境：

```text
Python executable = D:\Apps\Miniconda3\envs\env_tslib\python.exe
PyTorch = 2.5.1+cu124
CUDA available = true
GPU = NVIDIA GeForce GTX 1060, 6 GiB
```

未安装、复制或改写环境包。

## 3. Canonical Full 来源

唯一来源：

```text
D:\PaperProject\GyxPaper2\custom_models\results\st_mgprompt_component_ablation\component_ablation_p1_seed2026\A1\STMGPrompt_ComponentAblation
```

旧名称：

```text
旧 A1：w/o VADSP
```

旧实现的实际行为不是删除双分支，而是：

```text
use_vadsp = false
-> Fixed Dual
-> fine/coarse 两个分支均保留
-> gate 精确为 0.5 / 0.5
```

因此旧 A1 的实际配置与新 Canonical Full 一致：

```text
Fixed Dual
Bi-Diffusion
Last-State
hidden_dim = 64
num_coupling_layers = 1
granularity_weight_mode = difficulty_rate
site_weight_mode = dynamic
loss = msmg_dwu_loss
seed = 2026
```

历史 `config.json` 没有显式写入：

```text
vadsp_gate_mode
site_weight_mode
```

这两个字段由历史实现、`protocol_check.json`、训练日志和逐节点权重文件共同证明。该“历史隐式字段 -> 新 schema 显式字段”已记录在 `effective_config.json` 和 `provenance.json`，没有伪装成源 JSON 原本就存在的字段。

历史图算子名：

```text
bidirectional_diffusion
```

正式显示名：

```text
bi_diffusion
```

两者只做非语义名称映射；Canonical 的源 `config.json` 字节保持不变。

## 4. Canonical 兼容性与 hash

当前构造器只读复核：

```text
model state_dict strict=True：通过
model missing keys：0
model unexpected keys：0
loss state_dict strict=True：通过
loss missing keys：0
loss unexpected keys：0
参数量：278,534
预期参数量：278,534
```

关键 SHA256：

```text
config.json
19e3926e3c55f4ccaaf9f3d786807c706b26cd57050443521d42598854a18e56

best_checkpoint.pt
f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a

metrics.csv
ffc0d804e17a21a686038ed2d256af1cf74795c0aa81a05b16a57d91839506ea

metrics_eval_h3.json
881be8f8b56d7d1aa4515ec717b6d2ad591516062fbf31d9ec405bae354d0225

metrics_eval_h6.json
87f6b70246f1d9e97db10b638b164ef92f9d14f502da8d2829c94faf398bc633

metrics_eval_h10.json
c0f47960bdee49dbf92727a463784943c00c917cf6d00a18c3651f78df5a9c13
```

A0、P0 的 `reference.json` 均解析到同一 Canonical 目录，checkpoint/config/metrics hash 与上述值一致。

## 5. Canonical 迁移清单

Canonical 目录：

```text
D:\PaperProject\GyxPaper2\custom_models\results\st_mgprompt_canonical\full_fixed_dual_keep_msmgdwu_seed2026
```

从旧 A1 逐文件复制并逐文件验证 SHA256 的 18 个真实文件：

```text
config.json
active_config.json
best_checkpoint.pt
metrics.csv
metrics_eval_h3.json
metrics_eval_h6.json
metrics_eval_h10.json
model_summary.json
prediction_metadata.json
protocol_check.json
train_log.csv
train_complete.json
evaluation_complete.json
run_status.json
diagnostics/granularity_weight_curve.png
diagnostics/loss_diagnostics_summary.json
diagnostics/loss_weights.csv
diagnostics/site_weight_final.csv
```

新增派生文件：

```text
effective_config.json
canonical_manifest.json
artifact_hashes.json
provenance.json
canonical_verification.json
```

源目录不存在以下可选文件，因此未伪造：

```text
predictions.npz
metrics_val.json
metrics_test.json
eval.log
```

## 6. 为什么旧 P0-P5 不能复用

旧 P0-P5 建立在 Dynamic VADSP 完整设置上。新 P0-P5 必须全部以 Fixed Dual 为基础，因此：

```text
新 P0：只引用 Canonical Full
新 P1-P5：必须重新训练
旧 P0-P5 数值：不得进入新汇总
```

## 7. 为什么新 A1-A8 必须重跑

旧组件消融以旧 Dynamic VADSP A0 为基准，编号与新定义也不同。旧 A1 只能作为 Canonical Full 来源；新 A1-A8 均必须相对 Fixed Dual Canonical Full 做指定消融并重新训练。

## 8. A0/P0 为什么只是 alias

```text
CANONICAL_FULL = A0 = P0
```

三者是同一次训练、同一 checkpoint、同一 metrics。A0/P0 目录只保存 `reference.json`，不保存独立 checkpoint，也不手工复制 metrics 数值。

## 9. 最终 P0-P5 配置矩阵

| 变体 | 图算子 | 解码器 | hidden | coupling | 训练 |
| --- | --- | --- | ---: | ---: | --- |
| P0 | Bi-Diffusion | Last-State | 64 | 1 | 否，引用 Canonical |
| P1 | Simple | Last-State | 64 | 1 | 是 |
| P2 | Simple | Full-History Cross-Attn | 64 | 1 | 是 |
| P3 | Bi-Diffusion | Full-History Cross-Attn | 64 | 1 | 是 |
| P4 | Bi-Diffusion | Full-History Cross-Attn | 96 | 1 | 是 |
| P5 | Bi-Diffusion | Full-History Cross-Attn | 96 | 2 | 是 |

机器可读矩阵：

```text
custom_models/results/st_mgprompt_precision/precision_ablation_fixed_dual_seed2026/variant_config_matrix.csv
```

## 10. 最终 A0-A8 配置矩阵

| 变体 | 正式含义 | 唯一变化 |
| --- | --- | --- |
| A0 | Canonical Full | reference only |
| A1 | w/o Spatial Graph | 不执行节点间空间传播 |
| A2 | w/o Adaptive Graph | 关闭 adaptive graph，只用固定 prior |
| A3 | w/o Diffusion | Bi-Diffusion -> Simple |
| A4 | w/o Macro Prompt | 保留 coarse branch，关闭 Macro Prompt |
| A5 | w/o Reverse Cross | 关闭 fine -> coarse，保留 coarse -> fine |
| A6 | w/o Cross Fusion | 两分支独立编码，关闭双向融合 |
| A7 | w/o ST Prompt | 使用普通直接多步预测头 |
| A8 | w/o MS-MG-DWU | loss 改为 masked_score_aligned_hybrid |

机器可读矩阵：

```text
custom_models/results/st_mgprompt_component_ablation/component_ablation_fixed_dual_seed2026/variant_config_matrix.csv
```

A9、A10 会返回：

```text
Obsolete component-ablation variant. Valid variants are A0-A8.
```

## 11. 已修改或新增的主要文件

```text
custom_models/src/st_mgprompt/experiment_protocol.py
custom_models/src/st_mgprompt/config.py
custom_models/src/st_mgprompt/run_st_mgprompt.py
custom_models/src/st_mgprompt/formal_runner.py
custom_models/src/st_mgprompt/run_ablation.py
custom_models/src/st_mgprompt/run_precision_ablation.py
custom_models/src/st_mgprompt/canonical_artifact.py
custom_models/src/st_mgprompt/check_protocol.py
custom_models/src/st_mgprompt/registry.py
custom_models/src/st_mgprompt/test_component_ablation.py
custom_models/src/st_mgprompt/full_shape_matrix_smoke.py
custom_models/src/st_mgprompt/generate_delete_manifest.py
RUNBOOK_FIXED_DUAL_REFACTOR.md
HANDOFF_FIXED_DUAL_REFACTOR.md
DELETE_MANIFEST_refactor.json
DELETE_MANIFEST_refactor_bytecode.json
DELETE_STATUS_refactor.json
```

`formal_runner.py` 现在：

```text
P0/A0 使用 REFERENCE_ONLY
smoke 账本写入 results_smoke
smoke 状态不会冒充正式 COMPLETED
dry-run 生成正式 Full 命令但不执行
effective config 出现意外差异时阻止训练
汇总时解析 canonical reference
```

## 12. 已删除文件

已按主删除清单逐个删除 8 个 supplementary-only 源码文件：

```text
custom_models/src/st_mgprompt/run_supplementary_ablation.py
custom_models/src/st_mgprompt/test_supplementary_ablation.py
custom_models/src/st_mgprompt/fixed_dual_static_mg_candidate.py
custom_models/src/st_mgprompt/run_fixed_dual_static_mg_candidate.py
custom_models/src/st_mgprompt/test_fixed_dual_static_mg_candidate.py
custom_models/src/st_mgprompt/run_volatility_group_analysis.py
custom_models/src/st_mgprompt/volatility_group_analysis.py
custom_models/src/st_mgprompt/test_volatility_group_analysis.py
```

另按独立 bytecode 清单逐个删除上述模块的 8 个孤立 `.pyc` 文件。

总计已删除：

```text
16 个文件
```

删除均为单一明确路径；未使用递归、通配符或批量删除命令。

以下共享模块明确保留：

```text
volatility_patching.py
graph_prior.py
graph_layers.py
losses.py
decoder.py
cross_fusion.py
```

## 13. 尚未删除的旧结果

主删除清单生成时共有 996 个条目：

```text
DELETE_MANIFEST_refactor.json
```

全量校验结果：

```text
entry_count = 996
missing_count = 0
duplicate_count = 0
hash_mismatch_count = 0
outside_project_count = 0
migrated old-A1 entries = 19
```

8 个源码条目已删除，仍有 988 个旧结果 artifact 存在。项目 `AGENTS.md` 规定：需要批量删除时必须停止并请用户手动删除。因此没有用循环、递归、通配符或脚本批量删除这 988 个文件。

状态文件：

```text
DELETE_STATUS_refactor.json
```

剩余文件只属于以下已明确作废的旧结果范围：

```text
旧 Dynamic-VADSP A0-A10
旧 Dynamic-VADSP P0-P5
旧 method-full P0-P5
旧 method-full test P0-P5
旧 supplementary smoke
旧 Static-MG candidate
旧 volatility supplementary
```

用户需要按 `DELETE_MANIFEST_refactor.json` 中的明确路径手动逐项删除。Canonical、新正式目录和其他用户结果不在删除范围内。

## 14. 测试结果

### 静态与协议

```text
compileall：通过
关键模块 import check：通过
集中协议单元测试：7/7 通过
precision CLI help：通过
component CLI help：通过
A9 obsolete rejection：通过
A10 obsolete rejection：通过
完整 precision matrix：P0-P5
完整 component matrix：A0-A8
正式目录 checkpoint/metrics：0
```

### 普通 smoke

隔离目录：

```text
custom_models/results_smoke/st_mgprompt_fixed_dual_refactor
```

结果：

```text
P1-P5：全部通过
A1-A8：全部通过
```

每个变体均覆盖训练、验证、最佳 checkpoint、重载测试和协议检查；这些结果不进入正式结果汇总。

### Full-shape preflight

统一正式形状：

```text
B = 32
N = 134
C = 16
lookback = 144
max_pred_len = 10
validation batch = 4
```

机器可读报告：

```text
custom_models/results_smoke/st_mgprompt_fixed_dual_refactor_full_shape/full_shape_smoke_summary.json
```

报告完整性：

```text
required variants = 12
missing required variants = 0
passed = true
```

| 变体 | 设备/执行 dtype | 结果 |
| --- | --- | --- |
| P1 | CUDA float16 autocast | 通过 |
| P2 | CPU FP32 | 通过 |
| P4 | CPU FP32 | 通过 |
| P5 | CPU bfloat16 autocast | 通过 |
| A1-A8 | CUDA float16 autocast | 全部通过 |

所有变体均执行：

```text
forward
loss
backward
optimizer step
evaluation shape check
```

P5 的 CPU FP32 进程曾因系统级内存终止；保持完整正式形状、模型结构、loss 和 optimizer step 不变，改用显式记录的 CPU bfloat16 autocast 后通过。没有降低 B、没有使用 micro-batch、没有启动 epoch 训练。

P3 不在原指令的 full-shape 必测列表中；其普通 smoke 已通过。

## 15. 正式训练状态与下一步

尚未执行：

```text
P1 正式训练
P2 正式训练
P3 正式训练
P4 正式训练
P5 正式训练
A1 正式训练
A2 正式训练
A3 正式训练
A4 正式训练
A5 正式训练
A6 正式训练
A7 正式训练
A8 正式训练
```

正式运行命令与 PyCharm/Linux/nohup/自动关机说明见：

```text
RUNBOOK_FIXED_DUAL_REFACTOR.md
```

正式顺序：

```text
1. 解析 Canonical Full
2. 创建/刷新 P0 reference
3. 训练 P1-P5
4. 汇总 P0-P5
5. 创建/刷新 A0 reference
6. 训练 A1-A8
7. 汇总 A0-A8
```

## 16. 绝对不能再踩的坑

```text
不要重新训练 Canonical Full。
不要重新训练 P0 或 A0。
不要为 P0/A0 复制 checkpoint。
不要手工复制 canonical metrics 数值。
不要复用旧 P0-P5 数值。
不要复用旧 A0、A2-A10 数值。
不要把 A3 做成 w/o Trend Prior。
不要把 A4 做成删除 coarse branch。
不要把 A5 和 A6 混为同一消融。
不要把 A8 做成 Static-MG。
不要修改数据协议、checkpoint 选择标准、batch 或随机种子。
不要把 smoke 的 COMPLETED 状态冒充正式训练完成。
不要用递归、通配符、循环或批量命令删除剩余 988 个旧结果文件。
不要猜测 Linux 项目路径；先在服务器上用 pwd 和 test -f 验证。
```
