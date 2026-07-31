# E3-B graph identity integration audit

Frozen identities:

- graph ID: `sdwpf_physical_knn_v1`
- graph protocol:
  `f8224287f2a41be0e67e808445ce9a43b9973e7b075d1e1b4768396d6f98e1ef`
- graph bundle:
  `388a87dcba4fa0b897ab9b40b546d844433a26c799d50f4a009d832c6e335f0e`
- node order:
  `1bcfaa0afafb0a6535b9d24129bb2ba260cfd1782b9b485d960e6cfe4af90c35`
- location source:
  `233a120f4db2b52402bf4dc7c90631ac9301d6af236a56c6a245e46beec1f236`
- selected k: 4

The Native Graph Adapter verifies `(B,144,134,16)`, exact ordered TurbID
1...134, model/bundle identity, support names, and support hashes before every
forward. Runtime metadata mismatches fail closed; no automatic reorder occurs.

Resolved/effective config, protocol check, model summary, artifact manifest,
prediction metadata, run status, checkpoint metadata, and graph-aware
hardware-preflight identity carry the graph identity. Strict checkpoint load
checks graph identity before `load_state_dict(strict=True)`. Preflight reuse
also checks graph, node-order, support, config, source-closure, shape, AMP, and
forward/backward completion. Non-graph preflight identity remains unchanged.
