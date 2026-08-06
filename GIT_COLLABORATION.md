# GyxPaper2 协作说明

## 克隆与 LFS

```powershell
git clone git@github.com:efanxu/GyxPaper2.git
Set-Location -LiteralPath .\GyxPaper2
git lfs install
git lfs pull
```

默认分支为 `main`。开发分支使用 `feature/<short-name>`、`fix/<short-name>` 或 `experiment/<short-name>`，提交 PR 合并，不使用强推覆盖共享历史。

## 数据准备

GitHub 不下载数据集。原始或处理后的数据主体放在本地 `dataset/` 中；项目中的 Parquet 数据文件按 `.gitignore` 忽略。轻量 schema、列名说明、风机位置元数据和清洗报告可以纳入 Git。运行前使用项目脚本检查本地数据路径。

检查数据没有被跟踪：

```powershell
git ls-files dataset
git check-ignore -v dataset\sdwpf_10min.parquet
```

第一条只应显示轻量元数据文件；第二条应显示忽略规则。

## 日常开发

更新前先检查本地修改：

```powershell
git status --short
git pull --ff-only origin main
```

提交前只添加明确的逻辑批次路径，避免 `git add .`；检查 `git diff --cached --stat`、`git diff --cached --name-only` 和 `git lfs status`。提交后推送分支并创建 PR。

当前导入没有选择任何需要 Git LFS 的 canonical checkpoint；大型 checkpoint、数组、日志和可重建结果只保存在本地。完整路径、大小、content record 和排除原因见 `GIT_IMPORT_AUDIT.md`、`git_import_inventory.csv` 与 `git_import_excluded_paths.txt`。
