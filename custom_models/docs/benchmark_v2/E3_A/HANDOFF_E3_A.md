# HANDOFF E3-A

E3-A is complete and stops before E3-B.

`graph_protocol_v1` is frozen as `sdwpf_physical_knn_v1` for 134 integer TurbID nodes. It uses Cartesian x/y Euclidean distance, horizontal position only, deterministic minimum-connected k=4, Gaussian distance weights, explicit directed/undirected base matrices, GCN support, forward/reverse diffusion supports, and symmetric/scaled Laplacians.

Graph protocol hash: `f8224287f2a41be0e67e808445ce9a43b9973e7b075d1e1b4768396d6f98e1ef`

Graph bundle hash: `388a87dcba4fa0b897ab9b40b546d844433a26c799d50f4a009d832c6e335f0e`

Node-order hash: `1bcfaa0afafb0a6535b9d24129bb2ba260cfd1782b9b485d960e6cfe4af90c35`

STCN/STGCN is now the future canonical `stgcn` slot. No implementation was found or added, so it remains `MISSING_IMPLEMENTATION / BLOCKED_UNIMPLEMENTED`.

Registry remains 28 entries with 19 trainable, 9 locally full-shape verified, 10 preflight-required, 2 available non-trainable, and 7 blocked/unavailable. TSLib allowlist remains 18.

Tests: focused 25/25 PASS; full 121/121 PASS; compileall, Protocol check, graph-protocol check, double rebuild, and dummy contract PASS.

Formal training/evaluation, target-machine preflight, every formal graph implementation, and E3-B are NOT_RUN/NOT_STARTED.

E3-B may consider only GCN, source-verified classic STGCN, and DCRNN. GCN/STGCN use `A_gcn` (STGCN may additionally consume `L_tilde` only if its verified standard source requires Chebyshev support); DCRNN uses `P_forward` and `P_reverse`. No model may rebuild a separate location graph.
