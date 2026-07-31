# E3-B graph support usage audit

| model | supports | canonical hashes |
|---|---|---|
| GCN | `A_gcn` | `cb4e1b36e3200073425c2c263953a03e732007b5daff468992bc3783bc95a7f5` |
| STGCN | `L_tilde` | `c62790b9032d130389d356069f81ec3f819460e216cc9afb371d4a3b16b1f227` |
| DCRNN | `P_forward`, `P_reverse` | `59b39f001b3118557b55b88041b27b4ce7a743a85c0bc7948210945f7e3b8c03`, `89f09d41a8cae126c5668d4fe55934819c0a7489797b8e09fc0def0cc05da4fe` |

All supports come only from the frozen provider. Models do not read the
location CSV, compute distance, build KNN, alter k, learn a replacement common
graph, or import ST-MGPrompt graph layers. Frozen matrices remain float64 and
read-only; model-local runtime copies are float32, move explicitly with the
model, are `persistent=False`, require no gradient, and are absent from the
optimizer and checkpoint state dict.

GCN real-versus-identity, STGCN Chebyshev hand calculation, and DCRNN
dual-versus-identity diagnostics prove that graph support affects output.
