# E3-B configuration resolution

All configurations were frozen before the first ordinary smoke. Evidence order
was paper identity, author repository structure, the frozen benchmark task
shape, and the smallest explicit forecast adaptation. No validation search was
performed. Smoke/test results and local GTX 1060 memory outcomes were not used
for selection, and no OOM-driven capacity reduction occurred.

| model | frozen configuration | support |
|---|---|---|
| GCN | input 16; hidden 64; two graph layers; ReLU; dropout 0.1; shared `Linear(144,10)`; shared `Linear(64,1)` | `A_gcn` |
| STGCN | two ST-Conv blocks; `Kt=3`; `Ks=3`; each block `[64,16,64]`; GLU; graph ReLU; dropout 0.3; output collapse `136`, 64->128; LayerNorm/ReLU; shared 128->10 | `L_tilde`, frozen lambda-max policy 2.0 |
| DCRNN | encoder/decoder input 16/1; hidden 64; encoder/decoder layers 2/2; dual random walk; diffusion step 2; dropout 0; horizon 10; no teacher forcing/scheduled sampling/curriculum | `P_forward`, `P_reverse` |

For every model:

- `validation_search_performed=false`
- `test_result_used=false`
- `smoke_result_used_for_selection=false`
- `oom_driven_capacity_reduction=false`
- `clean_room_implementation=true`

The Benchmark Protocol optimizer, learning rate, weight decay, epochs, patience,
seed, AMP, loss, horizon, lookback, feature order, and split policy remain
authoritative and unchanged.
