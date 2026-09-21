# HANDOFF STEP6

## 状态

F0-F8 已注册到 `EMPIRICAL_ANALYSIS_V1`。F4/F5 的方向已按实际 query key value 流冻结：
F4 仅保留 Macro to Fine，F5 仅保留 Fine to Coarse。F1/F2/F3 不实例化 Macro Prompt 或
Cross-Attention；F3 的 gate 为节点 时间 通道级，范围为 `[0,1]`，初始 bias 为 0，默认不
stop-gradient。

- dry-run 唯一差异审计：已实现。
- 普通 smoke：已完成。
- `[32, 10, 134]` 正式形状前向/反向预检：已完成。
- F6/F7/F8：只读引用 A7 Canonical A4，不复制 checkpoint 或预测数组。
- 正式长训练与敏感性筛选：未启动；需要用户单独批准。

缺失的正式指标和诊断均保留为空或 `NOT_APPLICABLE`，没有补零。Cross-Fusion 权重仅解释为
历史表示层关联，不解释为未来 horizon 因果注意力。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step6_fusion.py -q
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --dry-run
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --smoke
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --full-shape
```

经用户批准后才运行正式单 seed 长训练和验证集敏感性筛选。
