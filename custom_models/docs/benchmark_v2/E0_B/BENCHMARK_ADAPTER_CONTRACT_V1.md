# Adapter contract v1

`BenchmarkBatch` requires `x (B,T,N,C)`, `target (B,N,H)`, `mask (B,N,H)`, stable sample IDs, window end indices, node IDs, split and metadata. `target_raw_or_inverse_transform` is separate from the model-space target. Metadata cannot carry future targets.

`BenchmarkOutput` requires exactly `prediction (B,N,10)`, finite values, `Patv_raw` semantic trace, raw shape and explicit semantics. Dimensions alone never establish target meaning.

`NodeSharedAdapter` performs the documented B/N reshape and rejects `(B*N,H,C)` without an explicit target-output policy. `NativeSpatiotemporalAdapter` accepts only native `(B,N,H)` output. A graph adapter requires `GraphContext` with node order, adjacency metadata, graph hash, directed/self-loop flags and normalization metadata.

