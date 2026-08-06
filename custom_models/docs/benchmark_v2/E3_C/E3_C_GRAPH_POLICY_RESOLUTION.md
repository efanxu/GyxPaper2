# E3-C graph policy resolution

Frozen graph context:

- graph: `sdwpf_physical_knn_v1`
- graph protocol: `<removed-content-record>`
- bundle: `<removed-content-record>`
- node order: `<removed-content-record>`
- location source: `<removed-content-record>`

| Model | Physical support enters forward | Learned graph/identity | Canonical policy |
|---|---:|---|---|
| Graph WaveNet | yes: `P_forward`, `P_reverse` | low-rank adaptive adjacency | three support branches, order 2 |
| MTGNN | no | directed top-k constructor | exactly 20 non-self outgoing edges per row |
| AGCRN | no | dense DAGG plus NAPL | bases are exactly `I`, `A` |
| STID | no | node/TOD/DOW embeddings | no adjacency or graph propagation |

`requires_graph=true` for all four means the runtime must validate the same
canonical 134-node GraphBundle and node order. It does not imply that a physical
matrix enters every forward pass. The adapter fails closed on reordered,
missing, extra, duplicate, boolean, or same-count-wrong node identities and on
graph identity mismatches.
