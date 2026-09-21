# HANDOFF STEP4

## 状态

G0–G8 的配置、真实前向差异、图来源隔离、自由与固定自适应支持、逐行度数保持随机重连及审计产物已实现。G0 只读引用 `full_fixed_dual_keep_msmgdwu_seed2026`。G1–G8 的 dry-run、普通 smoke 和正式形状前向/反向预检均通过；正式长训练未在本次预检阶段启动，G1–G8 保留 failure report，不填零或复制指标。

## 验证结果

- 步骤4合同测试：18 passed。
- 实证分析回归测试：44 passed，1 个 Matplotlib 参数弃用警告。
- 核心组件与 v4.3 回归测试：37 passed，1 skipped。
- G0–G8 dry-run：全部唯一差异审计通过。
- G1–G8 smoke：全部完成训练、评估与协议检查。
- G1–G8 full-shape：全部输出 `[32, 10, 134]`，loss 与梯度有限。
- 295 个 JSON 与 37 个 CSV 通过 Python 可读性检查。

## 已生成审计

- `G_VARIANT_MANIFEST.json`
- `G_PROTOCOL_AUDIT.json`
- `G_GRAPH_IDENTITY.csv`
- `G_BRANCH_ASSIGNMENT_TRACE.json`
- `G_SUPPORT_OVERLAP.csv`
- `G_EDGE_DISTANCE_SUMMARY.csv`
- `G_NODE_GROUP_METRICS.csv`
- `G_FAILURES.csv`

## 运行命令

```powershell
python -m pytest custom_models/tests/empirical_analysis/test_step4_graph_semantics.py -q
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --dry-run
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --smoke
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --full-shape
```

## 未完成项

- G1–G8 正式长训练与正式 H3/H6/H10、逐风机和场景指标。
- 多随机种子稳定性属于后续步骤11。

所有结论仅解释为统计语义匹配、先验支持集约束或模型内部边权稳定性，不解释为尾流或地理因果。
