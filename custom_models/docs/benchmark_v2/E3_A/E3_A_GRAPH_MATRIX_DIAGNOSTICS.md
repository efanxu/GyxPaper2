# E3-A graph matrix diagnostics

All matrices are `(134,134)`, finite, CPU float64, and record by canonical content.

| matrix | canonical record | symmetric | self-loop count | row-sum range | value range |
|---|---|---:|---:|---|---|
| A_binary_directed | `<removed-content-record>` | no | 0 | 4 to 4 | 0 to 1 |
| A_binary_undirected | `<removed-content-record>` | yes | 0 | 4 to 8 | 0 to 1 |
| A_directed | `<removed-content-record>` | no | 0 | 1.303874445837 to 2.281523331093 | 0 to 0.823103179261 |
| A_undirected | `<removed-content-record>` | yes | 0 | 1.303874445837 to 3.378149599739 | 0 to 0.823103179261 |
| A_gcn | `<removed-content-record>` | yes | 134 | 0.918094745678 to 1.152750414870 | 0 to 0.434051430974 |
| P_forward | `<removed-content-record>` | no | 0 | approximately 1 | 0 to 0.585446994928 |
| P_reverse | `<removed-content-record>` | no | 0 | approximately 1 | 0 to 0.689829019263 |
| L_sym | `<removed-content-record>` | yes | 134 | -0.286414396955 to 0.220866332786 | -0.464617506048 to 1 |
| L_tilde | `<removed-content-record>` | yes | 0 | -1.286414396955 to -0.779133667214 | -0.464617506048 to 0 |

The undirected base has one connected component, no isolated node, degree min/mean/median/max 4/4.4477611940/4/8, and density 0.03344181349. Full candidate-k, degree, distance, weight, symmetry, row-sum, finite, and self-loop diagnostics are in `graph_build_diagnostics.json`.
