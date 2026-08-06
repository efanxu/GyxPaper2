# E3-A implementation report

E3-A successfully froze `graph_protocol_v1` for the 134 SDWPF turbines without implementing or running a graph model.

The canonical node axis is integer `TurbID` 1..134 from the actual benchmark provider ordering rule. The neutral physical graph is a CPU float64 Cartesian x/y KNN graph. The first connected symmetrized candidate is k=4. It has 536 directed edges, 298 undirected edges, one connected component, no isolated nodes, no base self-loop, and sigma 921.5755719183301.

Frozen identities:

- node order: `<removed-content-record>`
- node metadata: `<removed-content-record>`
- graph bundle: `<removed-content-record>`
- graph protocol: `<removed-content-record>`

GraphSpec, read-only GraphBundle, explicit device transfer, frozen-only provider, fail-closed identity checks, checkpoint/artifact/preflight metadata rules, and native `(B,T,N,C)` contract are implemented. The CLI command `graph-protocol-check` loads and verifies without constructing a model or CUDA context.

STCN/STGCN was resolved to the future canonical `stgcn` slot, but no local implementation exists. Registry status is `MISSING_IMPLEMENTATION / BLOCKED_UNIMPLEMENTED`; train/evaluate remain false. The old unresolved ID remains a compatibility alias and `STCN` is only legacy inventory metadata.

Verification passed: compileall, 25 focused tests, 121 full tests, Protocol check, graph-protocol check, identical double rebuild, and CPU dummy contract smoke. No formal output root, formal train/evaluation, target-machine preflight, E3-B work, dependency change, file deletion, or file move occurred.
