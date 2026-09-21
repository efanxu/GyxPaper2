# 步骤0：已有实验资产审计报告

生成时间：2026-09-21T09:30:51.275724+00:00（只读审计；未修改历史实验文件，未生成哈希）。

## 1. 审计范围与协议

本次按 `PLAN.md` 检查 README、Fixed-Dual runbook/handoff、A0-A8 当前定义、协议/运行/评估/指标/诊断/预测导出源码，以及 `custom_models/results` 下 Canonical、P0-P5、A0-A8、E7/E8/E9、8 个正文基准模型和历史根目录。协议快照为：seed=2026，目标 `Patv_raw`，输入对齐信号 `Patv_clean_for_input`，history=96，正式 horizon=3/6/10，窗口 horizon=[6,18,36]。

输出文件：

- `existing_runs.csv`：逐运行/分析根的状态、指标、checkpoint、配置、诊断与预测导出清单。
- `existing_artifacts.json`：机器可读的来源、复用边界、阻塞项和 Step2 数据证据。
- `reuse_matrix.csv`：证据任务到现有资产的复用矩阵。
- `audit_report.md`：本报告。

## 2. Canonical 身份与别名去重

唯一 Canonical 训练根是 `custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026`。A0 和 P0 的 `reference.json` 都指向该目录，均为 reference-only，不是独立训练。正式准确率必须按 **1 个 Canonical 训练**计数，不能把 A0/P0 加成两个 run。当前 A6/A7/A8 的定义以 `A0_A8_FORMAL_DEFINITIONS.md` 为准：A6=early-history macro cross（排除最近 24 步）、A7=shared-projection cross-fusion、A8=w/o MS-MG-DWU（fair_main loss）。

## 3. 当前 Fixed-Dual 可复用情况

- Canonical/A0/P0：准确率可复用；A0/P0 仅别名。
- 组件消融：A2-A8 共 7 个当前正式 artifact 可做准确率聚合；A1 不可直接复用。
- 精度消融：P1-P3 共 3 个干净 artifact；P4/P5 的 checkpoint、训练/评估和 H3/H6/H10 文件存在，但桌面 `P0_P5.xlsx`/运行状态记录顶层 FAILED/Windows returncode，故只允许准确率复用，明确排除效率结论。
- 8 个正文基准：lightts、moving_average、tide、transformer、micn、timesnet、wpmixer、frets 均有当前 `benchmark_v2_uniform_bs4` formal artifact；可用于准确率、参数量和难度上下文，跨硬件效率仍需同机条件。

## 4. 必须阻断的语义/状态冲突

### A1

A1 目录中的 `config.json` 声明 `A1 / w/o Spatial Graph`，但 `active_config.json`、`requested_config.json`、`effective_config_diff.json` 和部分模型摘要落在 `A8 / w/o MS-MG-DWU` 语义；组件根状态还记录 `failed_protocol_mismatch`。因此 A1 数字文件只保留作历史/推理检查，不能作为当前 A1 正式消融或效率证据。需要重跑或完成一致性修复后才可纳入。

### P4/P5

P4/P5 的准确率结果可以复核，但运行状态冲突（桌面汇总的 Windows returncode 为 `3221225477`）意味着不能从它们得出效率结论。`reuse_matrix.csv` 已将两者拆成 accuracy-only。

### E7/E8/E9

E7 为 external 6/6 ready、internal graph 0/4；E8 为 external 6/6 ready、internal prompt/fusion 0/5。因此 E7/E8 只能作为外部只读诊断上下文。E9 readiness manifest 声称 6/6，但当前文件系统只有 5/6 个 transfer 目录（缺少：tide）；只有实际存在的行可复用，完整六模型结论需先对齐 manifest。另有 `E9_PAIRING_AUDIT.json` 的 exact initial-state pairing 为 NOT_VERIFIED，只能声称同 seed/source/config/protocol，不能声称逐张量同初始化。

历史 Dynamic-VADSP、旧 method-full、smoke/fix 根已列入 inventory，但均标为 `legacy_semantic_mismatch`，不进入当前 Fixed-Dual 正式表。

## 5. 现在可以直接开始的 no-training 分析

1. Step2 `data_evidence` 的对齐、train-only 统计、difficulty 分层和 formal metric-match 图表/表格。
2. Canonical + A2-A8 + P1-P3 的当前准确率汇总（A1、P4/P5 按上述规则排除相应结论）。
3. 8 个正文基准的准确率/参数量/难度上下文。
4. E9 transfer/control 准确率比较（当前 5/6 个目录可用；保留 exact-init 限制，先对齐缺失行）。
5. E7/E8 external-only 图结构及 prompt/fusion 诊断上下文；内部 grouped claim 等待 readiness gate。
6. 本次清单和复用矩阵本身的下游汇总；无需训练或加载 checkpoint。

## 6. 接受检查

| 检查项 | 结果 |
|---|---|
| 四个 Step0 文件存在 | ✅ 已写出并随后做 JSON/CSV 可读性检查 |
| Canonical 唯一身份可追溯 | ✅；A0/P0 去重 |
| A0-A8 当前定义已固定 | ✅；A6/A7/A8 采用当前正式定义 |
| A1 状态冲突被显式阻断 | ✅；不可用于正式 A1/效率 |
| P4/P5 accuracy-only 边界 | ✅；效率排除 |
| E7/E8 readiness 边界 | ✅；仅 external-only |
| E9 exact-init 限制 | ✅；NOT_VERIFIED 已保留 |
| 历史 Dynamic-VADSP 资产隔离 | ✅；标记 semantic mismatch |
| 哈希/历史文件修改 | ✅ 未生成哈希，未修改历史结果 |

结论：步骤0的**审计交付物**已完成且可机器读取；但这不等于所有历史/当前候选都已经“干净可用于全部结论”。A1、P4/P5、E7/E8 的限制仍必须遵守。步骤1/2的最终复核见主任务回复。
