# E3-A existing graph implementation audit

The workspace contained no pre-E3-A machine-readable public graph protocol that was frozen for every external graph benchmark. The detailed inventory is `existing_graph_implementation_matrix.json`.

Four identities were kept separate:

- A: 134-turbine physical or method graphs. The cleaning KNN is an imputation helper, while ST-MGPrompt macro/micro/adaptive graphs are private method components.
- B: MSGNet's adaptive `(16,16)` within-turbine variable graph.
- C: TimeFilter's sample-dependent `(B*N,8,144,144)` variable-time token graph.
- D: ST-MGPrompt private trend, weather, distance, adaptive, and diffusion graphs.

The cleaning helper uses fixed `k=5`, inverse distance with epsilon, and no frozen adjacency/hash. ST-MGPrompt's macro graph uses train-only `Patv_raw` and target mask; its micro graph uses train-only `Wspd`; its learned graph depends on model embeddings. None qualifies as a neutral, location-only common benchmark graph.

MSGNet and TimeFilter are not turbine graphs and were not copied. ST-MGPrompt sources and historical graph artifacts were read-only and were not modified.
