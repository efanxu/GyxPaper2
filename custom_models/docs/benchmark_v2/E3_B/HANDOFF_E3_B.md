# HANDOFF E3-B

E3-B is complete and stops before E3-C.

GCN, classic IJCAI-2018 STGCN, and DCRNN are integrated as native 134-turbine
graph models. All use only the frozen `sdwpf_physical_knn_v1` GraphBundle and
all passed ordinary, exact local full-shape, limited real SDWPF, strict reload,
artifact, leakage, batch isolation, node identity, and gradient checks.

Final statuses:

- GCN: `AVAILABLE_TRAINABLE`, exact local PASS.
- STGCN: `AVAILABLE_TRAINABLE`, exact local PASS.
- DCRNN: `AVAILABLE_TRAINABLE`, exact local PASS.

Registry: 28 total, 22 trainable, 12 locally exact PASS, 10 preflight-required,
2 non-trainable, 4 blocked. TSLib allowlist: 18.

Use `E3_B_RUNBOOK.md`. Formal Full, formal evaluation, and target-machine
preflight are NOT_RUN. The formal root
`custom_models/results/benchmark_v2/e3_b_seed2026` was not created.

The only next-stage inputs are Graph WaveNet, MTGNN, AGCRN, and STID. E3-C is
NOT_STARTED.
