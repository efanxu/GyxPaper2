# E3-B graph-model fairness audit

No smoke loss is used in this structural comparison.

- GCN uses the fixed 134-turbine physical graph, first-order symmetric
  normalized propagation, no dedicated temporal encoder, and a shared direct
  horizon projection.
- STGCN uses the fixed 134-turbine physical graph, Chebyshev spectral graph
  convolution, gated valid temporal convolution, and fully convolutional
  T-G-T blocks.
- DCRNN uses the fixed directed 134-turbine graph, forward/reverse random-walk
  diffusion inside DCGRU, an encoder-decoder, and autoregressive multi-step
  prediction.
- MSGNet's graph acts on 16 variables inside one turbine item; it is not a
  134-turbine graph.
- TimeFilter's graph acts on variable-time patch tokens; it is not a
  134-turbine graph.
- ST-MGPrompt uses private trend/weather/adaptive graphs and private diffusion;
  it is not a copy of `graph_protocol_v1`.

The three E3-B models do not import VADSP, Macro Prompt, Reverse Cross, Cross
Fusion, ST Prompt, MS-MG-DWU, or any ST-MGPrompt private graph layer.
