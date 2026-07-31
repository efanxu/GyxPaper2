# E2-A model specifications

All four models are node-shared:

`(B,T,N,16) -> (B*N,T,16) -> upstream -> single-target normalization -> (B,N,10)`.

One parameter set is shared across turbines. `cross_node_interaction=false`, `node_embedding=false`, `node_specific_head=false`; parameter count does not depend on `N`.

| model | parameters | single-target policy | ordinary | full-shape | limited real |
|---|---:|---|---|---|---|
| Transformer | 10,564,609 | native `c_out=1` predicts normalized `Patv_raw` | PASS | FAIL_OOM | PASS |
| PatchTST | 6,406,154 | dynamically select frozen `Patv_clean_for_input` channel to supervise normalized `Patv_raw` | PASS | FAIL_OOM | PASS |
| iTransformer | 6,385,162 | same dynamic power-channel policy | PASS | PASS, peak 4,015,753,728 B | PASS |
| TimeXer | 8,543,242 | native one-channel last-feature endogenous forecast | PASS | PASS, peak 3,224,407,552 B | PASS |

Transformer uses the last 48 observed history steps in its decoder prefix; its future 10 decoder steps and all required time marks are zero. PatchTST records original length 144, padding 8, padded length 152, 18 patches, and logical encoder patch shape `(B*N*16,18,512)`. iTransformer has exactly 16 variable tokens, representing features inside one turbine. TimeXer uses historical `Patv_clean_for_input` as endogenous and the other 15 historical features as exogenous.
