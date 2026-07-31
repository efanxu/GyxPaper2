# E3-B model specifications

All three models are native turbine-graph models with input
`(B,144,134,16)`, output `(B,134,10)`, one shared model instance, no
node-specific head, no node embedding, and no silent node reorder.

| model | parameters | architecture | fixed supports | final Registry status |
|---|---:|---|---|---|
| GCN | 6,763 | two first-order GCN layers plus shared direct horizon adaptation | `A_gcn` | `AVAILABLE_TRAINABLE` |
| STGCN | 1,169,002 | two classic T-G-T ST-Conv blocks and shared temporal-collapse output block | `L_tilde`, Chebyshev T0/T1/T2 | `AVAILABLE_TRAINABLE` |
| DCRNN | 385,793 | two-layer DCGRU encoder, two-layer DCGRU autoregressive decoder | `P_forward`, `P_reverse`, K=2 | `AVAILABLE_TRAINABLE` |

Runtime support copies are independent torch float32 non-persistent buffers.
Frozen matrices remain CPU float64, non-writeable, and unchanged. Supports are
not parameters and do not enter the optimizer.

Implementation paths:

- models: `custom_models/src/benchmark_v2/models/graph_models/`
- configs: `custom_models/src/benchmark_v2/configs/{gcn,stgcn,dcrnn}.py`
- adapter: `custom_models/src/benchmark_v2/adapters/e3_b.py`
- explicit builder: `custom_models/src/benchmark_v2/model_runtime.py`
