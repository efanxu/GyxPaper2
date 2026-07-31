# E3-A implementation report

E3-A successfully froze `graph_protocol_v1` for the 134 SDWPF turbines without implementing or running a graph model.

The canonical node axis is integer `TurbID` 1..134 from the actual benchmark provider ordering rule. The neutral physical graph is a CPU float64 Cartesian x/y KNN graph. The first connected symmetrized candidate is k=4. It has 536 directed edges, 298 undirected edges, one connected component, no isolated nodes, no base self-loop, and sigma 921.5755719183301.

Frozen identities:

- node order: `1bcfaa0afafb0a6535b9d24129bb2ba260cfd1782b9b485d960e6cfe4af90c35`
- node metadata: `c7b631159b900d3d51f125e9a4f01a4e432cda2cdd75f1ec6f14d7868850c727`
- graph bundle: `388a87dcba4fa0b897ab9b40b546d844433a26c799d50f4a009d832c6e335f0e`
- graph protocol: `f8224287f2a41be0e67e808445ce9a43b9973e7b075d1e1b4768396d6f98e1ef`

GraphSpec, read-only GraphBundle, explicit device transfer, frozen-only provider, fail-closed identity checks, checkpoint/artifact/preflight metadata rules, and native `(B,T,N,C)` contract are implemented. The CLI command `graph-protocol-check` loads and verifies without constructing a model or CUDA context.

STCN/STGCN was resolved to the future canonical `stgcn` slot, but no local implementation exists. Registry status is `MISSING_IMPLEMENTATION / BLOCKED_UNIMPLEMENTED`; train/evaluate remain false. The old unresolved ID remains a compatibility alias and `STCN` is only legacy inventory metadata.

Verification passed: compileall, 25 focused tests, 121 full tests, Protocol check, graph-protocol check, identical double rebuild, and CPU dummy contract smoke. No formal output root, formal train/evaluation, target-machine preflight, E3-B work, dependency change, file deletion, or file move occurred.
