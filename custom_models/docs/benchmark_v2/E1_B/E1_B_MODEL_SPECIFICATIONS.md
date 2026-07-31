# E1-B model specifications

## Common contract

所有模型只接收 `BenchmarkBatch.x (B,144,N,16)`。Adapter 进行 `(B,T,N,C)->(B,N,T,C)->(B*N,T,C)`，一个模型参数集共享给全部节点，输出恢复为 `(B,N,10)`。`node_semantics=node_shared`，`cross_node_interaction=false`，没有 node embedding、node-specific head 或图。

上游原始输出均为 `(B*N,10,16)`。Adapter 动态选择 `Patv_clean_for_input` 关联通道，监督目标为 normalized `Patv_raw`。目标、mask 和未来协变量不进入模型 forward。

| model | parameter count | key config | ordinary | full-shape | real-data | Registry |
|---|---:|---|---|---|---|---|
| DLinear | 2,900 | moving_avg=25, individual=false | PASS | PASS, peak 230,003,712 B | PASS | AVAILABLE_TRAINABLE |
| LightTS | 104,342 | chunk_size=8, num_chunks=18, d_model=512 | PASS | PASS, peak 2,760,148,992 B | PASS | AVAILABLE_TRAINABLE |
| TiDE | 1,804,777 | d_model=512, d_ff=2048, e/d=2/1, c_out=1 | PASS | FAIL OOM, peak allocated counter 13,014,100,480 B | PASS | BLOCKED_FULL_SHAPE_OOM |
| SegRNN | 1,583,874 | seg_len=2, d_model=512 | PASS | FAIL OOM, peak allocated counter 10,311,048,704 B | PASS | BLOCKED_FULL_SHAPE_OOM |

OOM peak counters may exceed physical 6 GiB because PyTorch records attempted/allocated demand during the failed operation; they are failure diagnostics, not usable capacity.

## Code paths

- Loader: `custom_models/src/benchmark_v2/upstream/tslib_loader.py`
- Wrappers: `custom_models/src/benchmark_v2/models/{dlinear,lightts,tide,segrnn}.py`
- Adapters: `custom_models/src/benchmark_v2/adapters/{dlinear,lightts,tide,segrnn}.py`
- Shared strict channel normalization: `custom_models/src/benchmark_v2/adapters/tslib_channel.py`
- Runtime/CLI: `model_runtime.py`, `model_cli.py`, `cli.py`

TiDE/SegRNN remain runnable only in `run_mode=smoke` for diagnostics. Formal `train` and evaluate support are disabled in Registry after exact full-shape OOM.
