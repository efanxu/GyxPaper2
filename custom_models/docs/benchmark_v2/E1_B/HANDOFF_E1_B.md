# HANDOFF_E1_B

## What E1-B did

E1-B implemented a controlled local Time-Series-Library path for DLinear, LightTS, TiDE and SegRNN inside `benchmark_v2`: explicit loader, frozen configs, wrappers, model-specific adapters, registry/runtime/CLI integration, tests, ordinary smoke, exact full-shape smoke, limited real SDWPF smoke, provenance artifacts and formal runbook.

No formal 20-epoch training/evaluation was started. E2-A was not started.

## Final model status

| model | source SHA256 | config | params | ordinary | full-shape | real | Registry |
|---|---|---|---:|---|---|---|---|
| DLinear | `e03acaf97e70c85ea6bda6776b3a62550f84470afa7bc53dffccb6121626e7eb` | ma=25, individual=false | 2,900 | PASS | PASS, 230,003,712 B | PASS | AVAILABLE_TRAINABLE |
| LightTS | `1de1583c4ee0bbfa0764fa9b158f09ac8bcd7ddf7f2d9f1b58ab1a03a7d13989` | chunk=8, chunks=18, d_model=512 | 104,342 | PASS | PASS, 2,760,148,992 B | PASS | AVAILABLE_TRAINABLE |
| TiDE | `4ab07dec4ae85f8b7c3062ff7d2fec00be342968d41cf09e48d599d8e40f6143` | d_model=512, d_ff=2048, e/d=2/1, c_out=1 | 1,804,777 | PASS | FAIL OOM | PASS | BLOCKED_FULL_SHAPE_OOM |
| SegRNN | `764d4ad950ee65162c70746248754942cfb4ba661162d515caab0a4eedd12252` | seg_len=2, d_model=512 | 1,583,874 | PASS | FAIL OOM | PASS | BLOCKED_FULL_SHAPE_OOM |

源码路径分别为 `Time-Series-Library/models/DLinear.py`、`LightTS.py`、`TiDE.py`、`SegRNN.py`。

Registry counts: total 28, available 5, unavailable/blocked 23, supports_train 3, non-trainable 2. TiDE/SegRNN 没有为了目标 7 被错误标为 available。

## Loader

`custom_models/src/benchmark_v2/upstream/tslib_loader.py` 的 allowlist 只有四项，不扫描目录，不接受任意 module string，不联网。它从自身位置解析项目根和 TSLib 根，检查精确文件，计算 SHA256，以私有 module name 加载；临时 sys.path 在 finally 中还原，并拒绝外部同名 `layers` 包。

Registry list/show 和 protocol-check 不导入 TSLib；model creation 才 lazy-load。resolved/effective/model summary 记录 source/license 绝对路径、hash、wrapper、adapter、source_modified=false。

## Input/output semantics

四模型均为：

```text
(B,T,N,16) -> permute/reshape -> (B*N,T,16)
-> upstream raw (B*N,10,16)
-> dynamic Patv_clean_for_input channel selection
-> (B*N,10) -> (B,N,10) normalized Patv_raw
```

一个参数集共享所有节点，无 cross-node interaction、node embedding 或 node-specific head。参数量不随 N 变化；相同历史产生相同预测，修改节点 A 不影响节点 B。

## TiDE fairness

TiDE 不使用 future target、future Patv_clean、future wind、ERA5、test target 或真实 time marks。Adapter 从历史 x 之外只构造零张量：`x_mark_enc=(B*N,144,4)`、future mark `(B*N,10,4)`、空 decoder placeholder。测试确认 target/mask 修改不改变 forward，所有 placeholders 为零，非零 future observed covariates 被拒绝。

## SegRNN

原始 `seg_len=96`：`seg_num_x=1`、`seg_num_y=0`；144 不能被 96 整除，reshape 破坏 `B*C`，预测批维最终为 0，与 `seq_last` 批维相加产生 dimension-0 mismatch。

`seg_len=2` 同时整除 144 和 10，得到 72/5 segments，raw output `(B*N,10,16)`。选择完全来自 shape constraint。Lookback、horizon 与 TSLib source 均未修改。

形状修复通过 ordinary/real smoke，但 exact `(32,144,134,16)`、AMP、d_model=512 在首段 GRU 处额外请求 4.71 GiB 并 OOM，所以正式状态 blocked。

## Tests and smoke

- unittest 46/46 PASS。
- protocol-check PASS。
- ordinary 4/4 PASS；backward、optimizer、validation H10 checkpoint、strict reload、artifact validation 均完成。
- full-shape DLinear/LightTS PASS；TiDE/SegRNN FAIL OOM，错误全文在 `full_shape_model_smoke_results.json`。
- limited real SDWPF 4/4 PASS；每模型最多 train/val/eval 2/1/1 batches；R2 无足够有效样本时写 JSON null。

所有 smoke 位于 `custom_models/results_smoke/benchmark_v2/e1_b`。指标仅用于诊断，不用于论文比较。

## Protection

- Protocol before/after: `0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`。
- Canonical before/after: `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`。
- ST-MGPrompt, Canonical tree, P0-P5/A0-A8 protected result trees, graph metadata: unchanged。
- TSLib models/layers/exp/data_provider/run.py/LICENSE: unchanged。
- protocol/data/loss/metrics/schemas/checkpointing/artifacts: unchanged。
- E1-A Persistence/MovingAverage/GRU model/config/adapters: unchanged。
- Dependencies changed: false；files deleted: none。
- Formal root existed before: false；exists after: false。

## Formal runbook and remaining work

Runbook: `custom_models/docs/benchmark_v2/E1_B/E1_B_RUNBOOK.md`。

Formal DLinear training: NOT_RUN。Formal LightTS training: NOT_RUN。Formal TiDE training: NOT_RUN/BLOCKED。Formal SegRNN training: NOT_RUN/BLOCKED。

TiDE/SegRNN 需要后续显式容量/硬件审计；不得静默降低 d_model、batch、N、T 或 horizon。若在更大 GPU 上重新执行 exact full-shape PASS，才可重新审计 Registry availability。

## E2-A input only

下一阶段仅为 Transformer、PatchTST、iTransformer、TimeXer。复用同一个显式 TSLib loader（扩展 allowlist 时逐项审计）、node-shared reshape、动态单目标输出协议、provenance fields、无未来信息约束、Trainer/Artifact/Smoke 路径。不得使用 TSLib auto-scan/trainer，不得自动开始 E2-A。
