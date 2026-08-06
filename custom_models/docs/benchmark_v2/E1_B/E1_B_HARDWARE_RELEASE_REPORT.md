# E1-B hardware release report

| model | old status | new status | supports train/evaluate | preflight required |
|---|---|---|---|---|
| TiDE | `BLOCKED_FULL_SHAPE_OOM` | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` | true / true | true |
| SegRNN | `BLOCKED_FULL_SHAPE_OOM` | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` | true / true | true |

The old GTX 1060 OOM evidence is permanently retained: `known_smoke_*`, `historical_local_full_shape_status`, complete OOM summary, peak-memory record, hardware identity, and `custom_models/docs/benchmark_v2/E1_B/full_shape_model_smoke_results.json`. That artifact and TiDE/SegRNN source/config/adapter records are protected and unchanged.

An RTX 4060 `train` request is no longer rejected by the historical Registry state. The parent launcher resolves the exact formal configuration and looks for a matching PASS. If missing, it starts a preflight child at `B=32,T=144,N=134,C=16,H=10`, AMP, forward, masked loss, and backward. The child exits completely. Only then does the parent start a new formal-training child from a clean CUDA context, which continues into epoch 1 without a second user command.

`FAIL_OOM` or `FAIL_NON_OOM` returns nonzero and prevents formal run creation. No batch/capacity reduction, gradient accumulation, node chunking, or Registry bypass is performed.

RTX 4060 preflight and all formal training remain NOT_RUN.
