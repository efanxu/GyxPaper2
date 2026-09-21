# HANDOFF STEP5

## 状态

D0-D5 已注册到 `EMPIRICAL_ANALYSIS_V1`。D0 使用 simple/local graph control 且扩散状态数为 0；D1、D2 分别使用 `A` 与 `A^T` 的一阶状态；D3、D4、D5 分别使用双向一阶、二阶和三阶递推。所有变体固定引用 `G0/CANONICAL` 图身份，转置方向只做 `row_normalize(A_transpose)`，不重新 TopK。

- dry-run 唯一差异审计：已实现。
- 普通 smoke：已完成。
- `[32, 10, 134]` 正式形状前向/反向预检：已完成。
- 正式长训练：未启动；需要用户单独批准。

## 验证结果

- 步骤5扩散合同测试：24 passed。
- 实证分析回归测试：62 passed，1 个 Matplotlib 参数弃用警告。
- 核心组件与 v4.3 回归测试：37 passed，1 skipped。
- D0、D1、D2、D3、D5 full-shape 均输出 `[32, 10, 134]`，loss 与梯度有限。
- full-shape 峰值显存从 D0 的 7556.01 MB 增至 D5 的 9911.15 MB；参数量与状态数已写入效率汇总。
- 201 个 JSON 与 26 个 CSV 已完成语法可读性检查。

## 产物

- `D_VARIANT_MANIFEST.json`
- `D_PROTOCOL_AUDIT.json`
- `D_OPERATOR_TRACE.csv`
- `D_STATE_COUNT_SUMMARY.csv`
- `D_PERFORMANCE_SUMMARY.csv`
- `D_SCENARIO_METRICS.csv`
- `D_EFFICIENCY_SUMMARY.csv`
- `D_FAILURES.csv`

未运行的正式指标保持为空并写入 failure report；没有补零、复制 checkpoint 或把矩阵方向解释为物理风向。

## 未完成项

- D0、D1、D2、D3、D5 的正式长训练与正式 H3/H6/H10、逐步、逐风机和场景指标。
- 多随机种子、配对 bootstrap 与 DM 检验属于后续步骤11。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step5_diffusion.py -q
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --dry-run
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --smoke
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --full-shape
```

经用户批准后才可运行正式长训练：

```powershell
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D5 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_diffusion `
  --run-full --skip-completed
```
