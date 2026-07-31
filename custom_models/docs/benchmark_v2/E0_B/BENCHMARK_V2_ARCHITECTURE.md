# benchmark_v2 architecture

`custom_models/src/benchmark_v2` is independent from ST-MGPrompt and TSLib. The package has one protocol loader, one explicit JSON registry, contracts, adapters, an SDWPF provider, mask-aware losses/metrics, checkpointing, artifacts, engine, runtime helpers and a single CLI.

Non-graph models use node-shared semantics: `(B,T,N,C) -> (B,N,T,C) -> (B*N,T,C)` with one shared parameter set, then `(B*N,H) -> (B,N,H)`. Native graph models keep `(B,T,N,C)` and require an explicit `GraphContext`; E0-B creates no graph.

Training accepts only `fit(train_loader, val_loader)`. Test evaluation is a separate evaluator after strict best-checkpoint reload. Smoke artifacts live below `custom_models/results_smoke/benchmark_v2`; future formal runs live below `custom_models/results/benchmark_v2`.

