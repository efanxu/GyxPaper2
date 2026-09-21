# HANDOFF STEP7

## 状态

N0-N8 已注册到 `EMPIRICAL_ANALYSIS_V1`，Prompt 固定为 `[1,H,N,D]`，预测统一为
`[B,H,N]`。node identity、horizon identity、shared horizon 和固定 input type embedding
均可独立审计。N1 使用零 horizon 分量，N3 使用一个可学习共享 horizon 向量，两者不是同一实现。

- dry-run 唯一差异审计：已实现。
- 普通 smoke：已完成。
- `[32,10,134]` 正式形状前向/反向预检：已完成。
- N5：只读引用 Canonical；N8：语义审计后只读引用 P3。
- N6/N7：只读取 144 步输入历史；N8 的 cross-attention key/value 仅来自同一历史窗口。
- 正式长训练：未启动；需要用户单独批准。
- prompt intervention：仅在 completed run 上显式执行，标签固定为内部敏感性分析，不是现实反事实。

## 修改文件

- `config.py`、`prompt_alignment.py`、`decoder.py`、`model.py`：独立 Prompt 开关与三类历史解码器。
- `empirical_protocol.py`、`run_empirical.py`、`check_protocol.py`：N0-N8 注册、运行、reference 和 fail-closed 审计。
- `step7_reporting.py`：步骤7清单、嵌入、解码器、性能、失败与交接产物。
- `scripts/empirical_analysis/prompt_intervention.py`：不改 checkpoint 的 inference-only 内部敏感性分析。
- `test_step7_prompt_decoder.py`：Prompt、decoder、前缀评价、干预恢复和唯一差异测试。

## 验证结果

- `compileall`：通过。
- ST-MGPrompt 核心与 empirical_analysis 回归：138 passed，1 skipped。
- N0-N8 dry-run：全部通过唯一差异审计。
- N0-N4、N6、N7 smoke：全部完成且 protocol passed；N5/N8 reference-only 审计通过。
- N0-N4、N6、N7 full-shape：全部完成，输出 `[32,10,134]`，loss 和 backward 有限。
- inference-only intervention：已在 N0 smoke checkpoint 上验证 zero、swap、mean；原参数在每次干预后恢复。

## 未完成项与下一步

正式长训练未启动，因此 N0-N4/N6/N7 的正式 H3/H6/H10 指标、Top10% 困难风机退化和
node embedding 相关分析仍标记为未开始。用户批准后运行正式单 seed 训练，再对 completed run
执行 prompt intervention 并刷新步骤7汇总。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step7_prompt_decoder.py -q
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --dry-run
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --smoke
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --full-shape
```
