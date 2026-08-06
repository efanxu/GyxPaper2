# AGCRN DAGG/NAPL trace

The shared node embedding has shape `(134,10)`. DAGG computes
`A=softmax(ReLU(E @ Eᵀ), dim=1)`. Chebyshev order 2 is resolved exactly as:

- `T0 = I`
- `T1 = A`

No `T2`, predefined adjacency, or physical support enters forward.

NAPL parameter-pool geometry:

| Recurrent cell | Combined input | Gate weight pool | Candidate weight pool |
|---|---:|---|---|
| layer 1 | 16+64=80 | `(10,2,80,128)` | `(10,2,80,64)` |
| layer 2 | 64+64=128 | `(10,2,128,128)` | `(10,2,128,64)` |

Bias pools follow the same embedding and basis axes. Tests verified nonzero
node-specific weight variation, both bases, both recurrent layers, the direct
10-horizon head, and gradients through node embeddings and every parameter
pool.

Ordinary DAGG was dense row-stochastic with 17,956 entries, 134 positive
self-loops, maximum asymmetry `0.00015133386477828026`, and record
`<removed-content-record>`.
Strict reload reproduced the record.
