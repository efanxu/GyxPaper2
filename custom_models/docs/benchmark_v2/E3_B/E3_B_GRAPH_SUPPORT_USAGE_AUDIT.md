# E3-B graph support usage audit

| model | supports | canonical records |
|---|---|---|
| GCN | `A_gcn` | `<removed-content-record>` |
| STGCN | `L_tilde` | `<removed-content-record>` |
| DCRNN | `P_forward`, `P_reverse` | `<removed-content-record>`, `<removed-content-record>` |

All supports come only from the frozen provider. Models do not read the
location CSV, compute distance, build KNN, alter k, learn a replacement common
graph, or import ST-MGPrompt graph layers. Frozen matrices remain float64 and
read-only; model-local runtime copies are float32, move explicitly with the
model, are `persistent=False`, require no gradient, and are absent from the
optimizer and checkpoint state dict.

GCN real-versus-identity, STGCN Chebyshev hand calculation, and DCRNN
dual-versus-identity diagnostics prove that graph support affects output.
