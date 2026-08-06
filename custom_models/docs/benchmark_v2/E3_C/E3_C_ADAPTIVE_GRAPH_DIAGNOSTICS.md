# E3-C learned graph and identity diagnostics

These values are engineering diagnostics from the single ordinary smoke
attempt. They are not frozen graph protocol artifacts, model-selection
evidence, or paper performance conclusions.

| Model | Learned identity record | Shape/density | Direction/self-loop | Reload |
|---|---|---|---|---|
| Graph WaveNet | `<removed-content-record>` | 134×134 / 1.0 | asymmetric / 134 positive | exact |
| MTGNN | `<removed-content-record>` | 134×134 / 0.1492537313 | directed / 0 | exact |
| AGCRN | `<removed-content-record>` | 134×134 / 1.0 | slightly asymmetric / 134 positive | exact |
| STID | `<removed-content-record>` | embedding 134×32 | no learned graph | exact |

For a common diagnostic-only row top-20 projection:

| Model | Edges | Out-degree | In-degree | Jaccard with `P_forward` | Jaccard with `P_reverse` |
|---|---:|---:|---:|---:|---:|
| Graph WaveNet | 2,680 | 20–20 | 4–42 | 0.0268199234 | 0.0248565966 |
| MTGNN | 2,680 | 20–20 | 0–59 | 0.0203045685 | 0.0219256435 |
| AGCRN | 2,680 | 20–20 | 4–40 | 0.0304389619 | 0.0301089045 |

The top-20 projection does not alter Graph WaveNet or AGCRN forward behavior.
For MTGNN it is the actual learned graph mask. Its 2,680 entries equal
`134×20`, with no self-loop. Graph WaveNet's full dense learned support has
Jaccard `0.0298507463` against either nonzero physical support because every
physical edge is contained in the dense mask.

STID's artifact records `learned_graph: null`, a node-embedding record, and a
diagnostic cosine-similarity matrix only. That similarity is not treated as an
adjacency.
