# HANDOFF E3-A

E3-A is complete and stops before E3-B.

`graph_protocol_v1` is frozen as `sdwpf_physical_knn_v1` for 134 integer TurbID nodes. It uses Cartesian x/y Euclidean distance, horizontal position only, deterministic minimum-connected k=4, Gaussian distance weights, explicit directed/undirected base matrices, GCN support, forward/reverse diffusion supports, and symmetric/scaled Laplacians.

Graph protocol record: `<removed-content-record>`

Graph bundle record: `<removed-content-record>`

Node-order record: `<removed-content-record>`

STCN/STGCN is now the future canonical `stgcn` slot. No implementation was found or added, so it remains `MISSING_IMPLEMENTATION / BLOCKED_UNIMPLEMENTED`.

Registry remains 28 entries with 19 trainable, 9 locally full-shape verified, 10 preflight-required, 2 available non-trainable, and 7 blocked/unavailable. TSLib allowlist remains 18.

Tests: focused 25/25 PASS; full 121/121 PASS; compileall, Protocol check, graph-protocol check, double rebuild, and dummy contract PASS.

Formal training/evaluation, target-machine preflight, every formal graph implementation, and E3-B are NOT_RUN/NOT_STARTED.

E3-B may consider only GCN, source-verified classic STGCN, and DCRNN. GCN/STGCN use `A_gcn` (STGCN may additionally consume `L_tilde` only if its verified standard source requires Chebyshev support); DCRNN uses `P_forward` and `P_reverse`. No model may rebuild a separate location graph.
