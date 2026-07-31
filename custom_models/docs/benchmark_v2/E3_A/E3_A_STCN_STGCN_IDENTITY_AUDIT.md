# E3-A STCN/STGCN identity audit

No local STCN or STGCN implementation was found. There is no class, model entry point, fixed adjacency consumer, graph convolution, temporal-convolution block, recurrent alternative, node embedding, source paper, license, checkpoint metadata, or old result artifact to classify.

Evidence:

- `custom_models/docs/benchmark_v2/E0_A/STCN_STGCN_NAMING_AUDIT.md`, SHA256 `58c2e096f80e0266394c1d672aca23757c1039f3d70fdb384c4ef236572bde92`
- `Time-Series-Library/run.py`, SHA256 `db6dfa215520ba193fe7095380f9005949a958efa9526975410cb5d97afe3fc4`; it exposes generic flags but constructs no STCN/STGCN

Final resolution:

- canonical ID: `stgcn`
- display name: `STGCN`
- legacy inventory alias: `STCN`
- compatibility alias: `stcn_stgcn_unresolved`
- identity: `RESOLVED_CANONICAL_TARGET_STGCN_SOURCE_MISSING`
- implementation: `MISSING_IMPLEMENTATION`
- runtime: `BLOCKED_UNIMPLEMENTED`
- legacy STCN: `UNVERIFIED_OR_DISTINCT_LEGACY`

This selects classic STGCN as the single future benchmark slot by policy; it does not claim that any historical STCN was STGCN. E3-B may perform source selection and implementation decision only. Training/evaluation remain blocked.

No matching historical STCN/STGCN artifact was found under results or results_smoke. Nothing was moved, renamed, deleted, promoted, or reused.
