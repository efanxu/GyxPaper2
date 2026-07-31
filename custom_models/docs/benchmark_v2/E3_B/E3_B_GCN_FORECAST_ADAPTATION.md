# E3-B GCN forecast adaptation

The model applies the same two Kipf-Welling first-order graph layers at every
historical time:

1. `(B,144,134,16) -> A_gcn @ X @ W1 -> (B,144,134,64) -> ReLU/dropout`;
2. `A_gcn @ H1 @ W2 -> (B,144,134,64) -> ReLU`;
3. permute to `(B,134,64,144)` and apply one shared `Linear(144,10)`;
4. permute to `(B,134,10,64)` and apply one shared `Linear(64,1)`;
5. return `(B,134,10)`.

There is no GRU, LSTM, temporal convolution, Transformer, adaptive adjacency,
node embedding, node-specific head, or cross-node dense forecast head.

Focused regressions verify the hand-calculated `A @ X @ W` formula, both graph
layers, temporal and output projection gradients, support non-trainability, and
different output under real `A_gcn` versus identity support.
