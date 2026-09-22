# 步骤 3–8 Windows 一次性运行与断点续跑

统一入口为：

```text
scripts/empirical_analysis/run_steps_3_to_8.ps1
```

该脚本按步骤 3、4、5、6、7、8 的顺序运行，覆盖：

1. Python 编译检查与六个步骤的专项测试；
2. 每个步骤的 dry-run、smoke、full-shape；
3. 每个步骤规定的单 seed 正式训练变体；
4. reference-only 变体只在预检阶段生成引用审计，不重新训练。

## 断点续跑

正式训练固定传入 `--resume --skip-completed`：

- 当前变体中断后，从其 `last_checkpoint.pt` 的下一 epoch 恢复；
- 已完整生成正式产物的变体直接跳过；
- 模型、优化器、损失状态、AMP scaler、早停计数与随机数状态均由训练器恢复。

脚本还会在 `custom_models/logs/steps_3_to_8_pipeline` 为完成的阶段写入 `.done`
标记。重新执行完全相同的命令时，已完成的编译、测试、dry-run、smoke、full-shape
或整步正式训练不会重复执行。若某阶段失败，不写标记；修复问题后重新执行原命令即可。

## 完整命令

从任意 PowerShell 工作目录执行：

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
& '.\scripts\empirical_analysis\run_steps_3_to_8.ps1' `
  -Mode All `
  -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' `
  -Seeds 2026 `
  -Device auto `
  -OutputRoot 'custom_models/results/empirical_analysis_v1' `
  -Profile 'STMG_FORMAL_V2'
```

同一条命令既是首次启动命令，也是中断后的续跑命令。

仅查看将要执行的命令，不运行：

```powershell
& '.\scripts\empirical_analysis\run_steps_3_to_8.ps1' -Mode All -PlanOnly
```

只运行预检或只运行正式训练：

```powershell
& '.\scripts\empirical_analysis\run_steps_3_to_8.ps1' -Mode Preflight
& '.\scripts\empirical_analysis\run_steps_3_to_8.ps1' -Mode Formal -Seeds 2026
```

多 seed 可一次传入，例如 `-Seeds 2026,2027,2028`。不同实验协议或 seed 集合应指定新的
`-StateDirectory`，避免沿用另一组运行的阶段标记。
