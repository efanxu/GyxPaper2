# E2-D batch and node isolation audit

All three models passed in `model.eval()`:

- batch companion invariance: fixed A gives the same output beside B or C;
- batch permutation invariance: permuting the batch only permutes output;
- single-item equivalence: A alone equals A inside a batch;
- cross-node isolation: changing turbine 0 leaves turbines 1+ unchanged;
- parameter-count invariance: rebuilding for different `N` does not change
  parameter count;
- one upstream model object, no node-specific head/embedding, no `(134,134)`
  graph.

Crossformer and TimeFilter have no batch aggregation in their upstream
forecast operations. MSGNet's rFFT top-k statistic explicitly averages batch,
so `(B*N)` cannot be passed as a shared frequency-selection batch. E2-D calls
the same model once for every flattened prediction item. Parameters are not
copied, and this is the adapter used by ordinary smoke, real smoke, exact
full-shape, preflight, and future formal training.

Target, mask, test target, and future target never enter forward. Supplied
calendar marks fail closed. Non-zero future observed covariates, exogenous
weather, or calendar tensors fail closed.
