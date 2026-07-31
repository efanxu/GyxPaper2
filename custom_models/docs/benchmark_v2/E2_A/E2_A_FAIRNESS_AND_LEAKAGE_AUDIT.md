# E2-A fairness and leakage audit

All forward paths consume only `BenchmarkBatch.x[:, -144:]`. The target, target mask, future observed covariates, future calendar covariates, test targets, and future exogenous tensors are excluded.

- Transformer: decoder prefix equals the last 48 observed `x` steps; the future 10-step decoder suffix is all zero. Encoder/decoder time marks are required by `DataEmbedding` and are all-zero placeholders. Changing target or mask leaves predictions bitwise unchanged.
- PatchTST: consumes historical input only. The upstream-designed end padding is fixed at 8; patch geometry is recorded and was not selected from smoke loss.
- iTransformer: adapter input is `(B*N,144,16)` and the token count is asserted to be 16. Turbines are flattened into batch and are never variable tokens.
- TimeXer: frozen feature-order hash is checked; index 15 must be `Patv_clean_for_input`. Indices 0–14 are historical exogenous features. Any future exogenous tensor is rejected, including an all-zero supplied tensor; non-zero future observed/exogenous input is fail-closed.

Tests also verify node independence, output `(B,N,10)`, dynamic power-channel selection, and target/mask invariance. No channel averaging, random channel, unchecked hard-coded target channel, future target, or high-capacity added decoder exists.
