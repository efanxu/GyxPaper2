# E3-A graph runtime contract

`GraphSpec` is immutable and records graph ID, schema, coordinate/distance/elevation policy, k rule and result, direction/self-loop/weight/normalization rules, and graph-protocol hash.

`GraphBundle` is immutable at the dataclass level; all NumPy matrices are float64 and non-writeable. It carries ordered node IDs, node/order/location identities, nine matrices, individual hashes, bundle hash, and protocol hash. Loading validates every frozen source and matrix fail-closed. It never rebuilds or overwrites.

Import and load are CPU-only and do not import torch. `bundle.to(device)` is the only tensor/device transition and returns a separate `GraphTensorBundle`.

The native contract is `(B,T,N,C)=(B,144,134,16)` to `(B,134,10)`. `validate_native_graph_input` checks the complete node axis and exact node order. Runtime graph protocol, node-order, and bundle hashes are explicit and must match. No silent reorder, missing-node fill, extra-node drop, or CSV-row-order dependence is allowed.

The E3-A dummy uses only x and GraphBundle in forward. Target and mask are not forward arguments. It is not registered and is not a benchmark model.
