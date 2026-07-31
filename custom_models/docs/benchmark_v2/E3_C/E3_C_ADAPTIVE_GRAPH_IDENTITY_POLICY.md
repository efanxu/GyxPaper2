# E3-C adaptive graph and identity policy

Learned graph parameters are ordinary model parameters: they are initialized
after the benchmark seed is set, participate in optimization, and are included
in strict checkpoints. They are not frozen graph protocol artifacts.

- Graph WaveNet owns independent `E1(134,10)` and `E2(10,134)` parameters.
- MTGNN owns two 40-dimensional node-embedding tables and constructs a directed
  top-k graph on every forward pass.
- AGCRN owns one `E(134,10)` table shared by DAGG and NAPL.
- STID owns a `(134,32)` node embedding but no learned adjacency.

Reusable checkpoint or hardware-preflight identity includes model ID, resolved
and effective config hashes, source-closure hash, benchmark protocol, exact
`B/T/N/C/H`, AMP, graph protocol, bundle, node order, location source, support
names/hashes, physical-support policy, adaptive-graph policy, node-identity
policy, temporal-identity policy, and adaptive initialization policy. Any
mismatch is rejected before state reuse.

Canonical learned-matrix hashes are diagnostic checkpoint-state identities, not
frozen protocol identities and not performance evidence.
