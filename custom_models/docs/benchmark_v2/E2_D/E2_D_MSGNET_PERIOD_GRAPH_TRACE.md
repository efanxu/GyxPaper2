# E2-D MSGNet period and variable-graph trace

`top_k=5`, `e_layers=2`. Each layer performs history-derived rFFT, retains all
five source branches, pads each branch to its period multiple, reshapes to
`(1,length/period,period,512)`, applies period-axis causal self-attention,
crops exactly with `[:, :144, :]`, softmax-weights all five branches, and adds
the residual.

Periods and weights are input-dependent. The adapter records for every
flattened item and layer:

```text
frequency_indices
periods = 144 // frequency_index
raw and softmax scale weights
per-period padding
FFT input/complex dtype
```

The diagnostic contract requires positive periods, five entries per layer,
FP32 rFFT statistics (`complex64` result), and no period deduplication or
branch removal. Multiple frequencies producing the same integer period remain
separate, matching upstream.

For the retained seed-2026 fixed-input diagnostic, layer 1 indices/periods are
`[1,16,2,3,8] / [144,9,72,48,18]` with softmax weights
`[0.743081,0.125206,0.070362,0.045608,0.015743]`. Layer 2 records
`[1,16,2,3,29] / [144,9,72,48,4]` with
`[0.550725,0.175818,0.133760,0.092613,0.047085]`. These are diagnostic
values, not frozen periods or selection criteria. Full values and gradients
are retained in
`custom_models/results_smoke/benchmark_v2/e2_d/diagnostics/e2_d_model_diagnostics.json`.

Every `GraphBlock` has `nodevec1=(16,10)`, `nodevec2=(10,16)`, and adaptive
adjacency `(16,16)`. Mix-hop depth is two. These nodes are the 16 embedded
within-turbine variables; no turbine-location file, graph provider, or
`(134,134)` adjacency is imported.

Upstream `frequency_list = abs(rfft(x)).mean(0).mean(-1)` couples its input
batch. E2-D isolates it with one shared model call per flattened prediction
item (`fft_batch_scope=1`). Batch companion, permutation, and single-item
equivalence regressions pass. All five graph branches in both layers receive
finite non-zero `nodevec1/nodevec2` gradients. Native AMP initially risks
non-power-of-two cuFFT behavior, so only FFT/statistics run in FP32; graph,
attention, aggregation, projection, and backward remain AMP-capable.
