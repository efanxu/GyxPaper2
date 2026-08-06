# E1-B implementation report

## Outcome

四个本地 TSLib 模型均完成 loader、wrapper、config、model-specific Adapter、统一 runtime/CLI、单元测试、ordinary smoke 和受限真实数据 smoke 接入。DLinear 与 LightTS 还通过正式 shape AMP backward，因此为 `AVAILABLE_TRAINABLE`。TiDE 与 SegRNN 在 6 GiB GTX 1060 上 exact full-shape OOM，按规则保持 `BLOCKED_FULL_SHAPE_OOM`。

最终 Registry：total 28，available 5，unavailable/blocked 23，trainable 3，non-trainable 2。没有为了目标数字伪造 available。

## Loader and provenance

`upstream/tslib_loader.py` 只有四项显式 allowlist，不枚举 `models/`，拒绝任意字符串。路径从项目根解析，临时 sys.path 在 finally 中完整恢复，并拒绝已加载的外部同名 `layers` 包。Registry list/show 和 protocol-check 不导入上游；create model 时才加载准确文件。effective/resolved/model summary 记录项目、绝对路径、content record、license、wrapper、adapter 与 source_modified=false。

## Semantics

四模型使用 node-shared B/N reshape。四个上游实现都返回 16 通道，Adapter 通过冻结 feature order/record 动态选择 `Patv_clean_for_input` 通道并监督 `Patv_raw`，拒绝错误 feature order 与错误 raw shape。

TiDE 只使用历史 x；historical/future time marks 与 x_dec 都由 Adapter 创建为零。修改 target 或 mask 不改变预测，非零 future observed covariates 被拒绝。

SegRNN 原 `seg_len=96` 错误已先复现并逐步追踪；`seg_len=2` 是纯协议整除决议。

## Verification

- Full unittest: 46/46 PASS；保留 E0-B 17 tests 与 E1-A 30-test 基线语义。
- masked raw NaN Score regression: PASS。
- protocol-check: PASS，record `<removed-content-record>`。
- Ordinary: 4/4 PASS，均 backward/optimizer/checkpoint/strict reload/artifact PASS。
- Full-shape: DLinear PASS，LightTS PASS，TiDE FAIL OOM，SegRNN FAIL OOM。
- Limited real SDWPF: 4/4 PASS；每模型最多 2/1/1 batches。

Smoke metrics 仅诊断，不进入论文比较。TiDE/SegRNN 的真实 smoke PASS 不覆盖其 full-shape FAIL。

## Protection and scope

未修改 TSLib、ST-MGPrompt、Canonical、E1-A 三模型、protocol/data/loss/metrics/schemas/checkpointing/artifacts。未安装依赖、未删除文件、未创建或写入正式结果根、未启动 20-epoch 训练、未进入 E2-A。

任务开始时无 Git metadata，因此不存在可声明的 branch/HEAD/clean status。
