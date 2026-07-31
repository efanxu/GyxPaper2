# E3-A graph matrix diagnostics

All matrices are `(134,134)`, finite, CPU float64, and hash by canonical content.

| matrix | canonical hash | symmetric | self-loop count | row-sum range | value range |
|---|---|---:|---:|---|---|
| A_binary_directed | `5101e2cd5affbde633b9d0a7c3193f475150b16cf5a285799070873f2db2290f` | no | 0 | 4 to 4 | 0 to 1 |
| A_binary_undirected | `6bdc5819e68396f6e61bd0306bfcc05ad8f08f2798031cc026ee8065b795019f` | yes | 0 | 4 to 8 | 0 to 1 |
| A_directed | `c07e3844092537f58dbd99363b60378c9e4ff64981437957506e818273dc69bf` | no | 0 | 1.303874445837 to 2.281523331093 | 0 to 0.823103179261 |
| A_undirected | `70a0c736d7bfe875e75f1f6279efb412db1b5f72248b34fc4f9bb5c7c9a19811` | yes | 0 | 1.303874445837 to 3.378149599739 | 0 to 0.823103179261 |
| A_gcn | `cb4e1b36e3200073425c2c263953a03e732007b5daff468992bc3783bc95a7f5` | yes | 134 | 0.918094745678 to 1.152750414870 | 0 to 0.434051430974 |
| P_forward | `59b39f001b3118557b55b88041b27b4ce7a743a85c0bc7948210945f7e3b8c03` | no | 0 | approximately 1 | 0 to 0.585446994928 |
| P_reverse | `89f09d41a8cae126c5668d4fe55934819c0a7489797b8e09fc0def0cc05da4fe` | no | 0 | approximately 1 | 0 to 0.689829019263 |
| L_sym | `300b77b20a0848bae9a1791ccd98a47ee2354c8fda4133870d7ca246bb6ad17a` | yes | 134 | -0.286414396955 to 0.220866332786 | -0.464617506048 to 1 |
| L_tilde | `c62790b9032d130389d356069f81ec3f819460e216cc9afb371d4a3b16b1f227` | yes | 0 | -1.286414396955 to -0.779133667214 | -0.464617506048 to 0 |

The undirected base has one connected component, no isolated node, degree min/mean/median/max 4/4.4477611940/4/8, and density 0.03344181349. Full candidate-k, degree, distance, weight, symmetry, row-sum, finite, and self-loop diagnostics are in `graph_build_diagnostics.json`.
