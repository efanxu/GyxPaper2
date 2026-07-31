# E3-C graph policy resolution

Frozen graph context:

- graph: `sdwpf_physical_knn_v1`
- graph protocol: `f8224287f2a41be0e67e808445ce9a43b9973e7b075d1e1b4768396d6f98e1ef`
- bundle: `388a87dcba4fa0b897ab9b40b546d844433a26c799d50f4a009d832c6e335f0e`
- node order: `1bcfaa0afafb0a6535b9d24129bb2ba260cfd1782b9b485d960e6cfe4af90c35`
- location source: `233a120f4db2b52402bf4dc7c90631ac9301d6af236a56c6a245e46beec1f236`

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
