# GyxPaper2

GyxPaper2 是一个面向风电时间序列预测与自定义模型实验的研究项目，包含模型实现、实验配置、运行脚本、测试、文档以及轻量级复现摘要。

## 项目结构

- `custom_models/src/`：自定义模型、训练与评估实现。
- `custom_models/tests/`：测试与轻量验证。
- `custom_models/docs/`：模型协议、实验说明和交接文档。
- `scripts/`：实验运行、预检查、汇总和快照脚本。
- `dataset/`：本地数据位置；数据主体被 Git 忽略，schema、元数据和清洗报告可协作。
- `Time-Series-Library/`：基础时间序列库及其运行组件。
- `GIT_IMPORT_AUDIT.md`：本次仓库导入的文件审计、SHA256 和排除原因。

## 数据准备

GitHub 不提供数据集下载。请将原始或处理后的数据放在本地 `dataset/` 目录，并按项目脚本要求配置路径。Parquet 数据主体不会被 Git 跟踪；可用 `git ls-files dataset` 和 `git check-ignore -v dataset\\sdwpf_10min.parquet` 确认。

数据列说明、风机位置元数据、清洗脚本和清洗报告可以提交，用于复现实验准备流程。

## 协作开发

克隆后执行 `git lfs install` 和 `git lfs pull`，再创建 `feature/<name>`、`fix/<name>` 或 `experiment/<name>` 分支。提交前先检查本地修改，只添加明确的逻辑批次并运行相应的轻量验证；不要使用 `git add .`。完整协作约定见 [`GIT_COLLABORATION.md`](GIT_COLLABORATION.md)。

本次导入没有选择 canonical checkpoint 或其他必须使用 Git LFS 的大文件。大型 checkpoint、数组、日志和可重建实验结果保留在本地；路径、大小、SHA256 与原因见 [`GIT_IMPORT_AUDIT.md`](GIT_IMPORT_AUDIT.md)。
