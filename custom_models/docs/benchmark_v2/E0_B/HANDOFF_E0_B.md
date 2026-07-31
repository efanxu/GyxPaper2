# HANDOFF_E0_B

E0-B froze the benchmark_v2 public runtime and stopped before E1-A. The only protocol source is `custom_models/src/benchmark_v2/protocol/benchmark_protocol_v1.json`; its hash is `0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`.

Package areas: `protocol/`, `registry/`, `contracts/`, `adapters/`, `data/`, `losses.py`, `metrics.py`, `checkpointing.py`, `artifacts.py`, `engine.py`, `runtime.py`, `cli.py`, and schemas. The registry has 28 explicit entries and no runtime-available model. The 18 TSLib sources are source-audited but not adapted; eight models are missing; GRU is only a partial internal block; STCN/STGCN remains unresolved.

Non-graph models use node-shared `(B,T,N,C)->(B*N,T,C)->(B,N,H)`. Graph models are native `(B,T,N,C)->(B,N,H)` and require explicit `GraphContext`; no graph is generated here. Future observed/calendar covariates are disabled. Default E1-E4 loss is masked MSE in normalized target space; E5 common loss comparison and E9 MS-MG-DWU are outside E0-B. Checkpoints monitor validation official Score H10, min, patience 6, min delta 0.01, and test is a separate post-training evaluator.

Smoke root: `custom_models/results_smoke/benchmark_v2`; formal root: `custom_models/results/benchmark_v2`. CLI: `PYTHONPATH=custom_models/src python -m benchmark_v2.cli registry-list`, `protocol-check`, `framework-smoke`, `full-shape-framework-smoke`, `artifact-validate --run-dir <path>`. No TSLib trainer or old result root is used.

Verification: the 17-test benchmark_v2 unittest suite is `PASS`; protocol check is `PASS`; ordinary framework smoke is `PASS` with artifact validation `PASS` and `(2,4,10)` output; full-shape framework smoke is `PASS` on CUDA with `(32,134,10)` output and recorded peak GPU memory; the real data-contract smoke is `PASS`, limited to two windows per split and no training. `custom_models/results/benchmark_v2` was not created and no formal run exists.

Protection: core and TSLib before/after snapshots have zero mismatches. Canonical SHA256 before and after is `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`. ST-MGPrompt core, TSLib sources, Canonical, P0-P5/A0-A8 and old result directories were not modified; no dependency was installed/upgraded/deleted and no file was deleted.

E0-A known smoke errors remain verbatim: SegRNN `RuntimeError: tensor dimension 0 mismatch` (E1-B); MICN `RuntimeError: embedding sequence dimension mismatch` (E2-B); MultiPatchFormer `RuntimeError: patch tensors have inconsistent lengths` (E2-B); TimeMixer `IndexError: season_list[1] does not exist` (E2-C). They were stored in registry metadata, not fixed.

Next step is only E1-A: implement Persistence, MovingAverage and an independent node-shared GRU through these contracts, with registry/config/adapter tests and isolated smoke artifacts. E1-A must not modify ST-MGPrompt, TSLib, Canonical, P0-P5/A0-A8, old runners, or the frozen protocol fields; must not use test during training, validation-only tuning, automatic scanning, remote download, or fake fallback models.
