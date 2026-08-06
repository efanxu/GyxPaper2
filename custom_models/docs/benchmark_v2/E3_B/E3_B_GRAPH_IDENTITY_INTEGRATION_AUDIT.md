# E3-B graph identity integration audit

Frozen identities:

- graph ID: `sdwpf_physical_knn_v1`
- graph protocol:
  `<removed-content-record>`
- graph bundle:
  `<removed-content-record>`
- node order:
  `<removed-content-record>`
- location source:
  `<removed-content-record>`
- selected k: 4

The Native Graph Adapter verifies `(B,144,134,16)`, exact ordered TurbID
1...134, model/bundle identity, support names, and support records before every
forward. Runtime metadata mismatches fail closed; no automatic reorder occurs.

Resolved/effective config, protocol check, model summary, artifact manifest,
prediction metadata, run status, checkpoint metadata, and graph-aware
hardware-preflight identity carry the graph identity. Strict checkpoint load
checks graph identity before `load_state_dict(strict=True)`. Preflight reuse
also checks graph, node-order, support, config, source listing, shape, AMP, and
forward/backward completion. Non-graph preflight identity remains unchanged.
