# HANDOFF STEP8

## 状态

L0-L7 已按 `EMPIRICAL_ANALYSIS_V1` 注册。L0-L6 共享 Canonical 模型与 Smooth L1
基础损失，仅改变已声明的 horizon/node 权重机制；L3 只读引用 Canonical。L7 只读审计
独立 Batch4 A8，不进入 L0-L6 的 32/4/4 组件因果表。

- dry-run 唯一差异审计：通过。
- 普通 smoke：以 `source=smoke_only` 写入权重和 loss 诊断，不作为论文性能数值。
- `[32,10,134]` full-shape：由各变体 `model_summary.json` 审计。
- checkpoint loss/optimizer/AMP/early-stopping/RNG 字段恢复审计：通过。
- A8 dedicated Batch4 readiness：NOT_READY；缺失或不完整时 fail closed。
- 正式长训练：未启动，需用户单独批准。

## 实现要点

- L0：static/static，权重恒为 1，等价 masked Smooth L1 reduction。
- L1：difficulty-rate/static；L2：static/dynamic；L3：Canonical reference。
- L4：训练前冻结 1:2:3 horizon 权重并归一化；L5：可学习 `log_sigma_g`。
- L6：仅在 epoch 结束时根据训练 loss 更新 DWA；无有效 loss 时保持原状态。
- 全无效 mask 返回 `None`，训练器跳过 backward 和 optimizer step并累计计数。

## 产物与限制

步骤8的 manifest、协议审计、loss 输入合同、epoch loss、horizon/node 权重、clip 分布、
checkpoint 恢复审计、性能占位、困难风机占位和失败清单均位于本目录。正式训练尚未执行，
所以 L0/L1/L2/L4/L5/L6 的正式 H3/H6/H10 指标和 Top10% 困难风机结果保持 `not_started`，
没有填 0 或复制 smoke 数值。

## 验证结果

- `compileall`：通过。
- 步骤8专项与 empirical/core 回归：143 passed，1 个既有 Matplotlib 弃用警告。
- L0-L7 dry-run：唯一差异审计全部通过。
- L0/L1/L2/L4/L5/L6 smoke：完成且 protocol passed；L3 为 Canonical reference。
- L0/L1/L2/L4/L5/L6 full-shape：输出均为 `[32,10,134]`，loss 与 backward 有限。
- L7：专用 A8 Batch4 路径不存在，按合同标记 `reference-unavailable`，没有读取旧 A8。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step8_losses.py -q
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --dry-run
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --smoke
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --full-shape
```
