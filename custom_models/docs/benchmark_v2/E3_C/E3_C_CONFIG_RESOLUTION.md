# E3-C frozen configuration resolution

All values below were fixed before smoke execution. Smoke results were not used
to choose or reduce capacity.

| Model | Frozen architecture |
|---|---|
| Graph WaveNet | input 16, horizon 10, dropout 0.3, residual/dilation 32, skip 256, end 512, kernel 2, 4 blocks × 2 layers, diffusion order 2, adaptive rank 10 with seeded random initialization, `P_forward` + `P_reverse` + adaptive support |
| MTGNN | GCN depth 2, dropout 0.3, top-k 20, node dim 40, 3 layers, dilation exponent 1, conv/residual 32, skip 64, end 128, propalpha 0.05, tanhalpha 3, affine layer norm; no physical support and no node split/sampling |
| AGCRN | embedding 10, recurrent hidden 64, 2 recurrent layers, Chebyshev order 2 interpreted exactly as `I` and adaptive `A`; DAGG + NAPL; no physical support |
| STID | series/node/TOD/DOW embeddings 32 each, TOD size 144, DOW size 7, 3 residual MLP blocks, dropout 0.15; only the final observed historical timestamp; no graph propagation |

Common contract: `(B,144,134,16) -> (B,134,10)`, canonical TurbID order
`1..134`, Adam at `1e-3`, masked MSE in normalized target space, seed 2026,
AMP enabled by the benchmark protocol, and no future observed/calendar/target
inputs.

The frozen config modules are under
`custom_models/src/benchmark_v2/configs/`. The closure records are:

- Graph WaveNet: `<removed-content-record>`
- MTGNN: `<removed-content-record>`
- AGCRN: `<removed-content-record>`
- STID: `<removed-content-record>`
