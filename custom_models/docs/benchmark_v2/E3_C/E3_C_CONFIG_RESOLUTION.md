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
`custom_models/src/benchmark_v2/configs/`. The closure hashes are:

- Graph WaveNet: `cf401b35cd1f9455eb9653e2e04ef28f02c2e1673812dd4abb4a19bca7366eb8`
- MTGNN: `99ce4ec1be02f20d3f580ecdd363a3bc336d2d4f1a180d7f517cf9e91fe2c7d3`
- AGCRN: `638bbfd5e7a451df556b458a98ae87b35490bcf15b38a0152b0faf6a2a67a7be`
- STID: `34dc11ac13ade13d3eb83cfb8cde98acfacf5ad4b18d3e79d653d5486dbe4bb9`
