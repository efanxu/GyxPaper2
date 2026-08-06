# Uniform Batch=4 Implementation Report

The versioned profile `uniform_train_batch4_v1` is implemented with record `<removed-content-record>`.
It injects train/val/test batch 4, gradient accumulation 1, effective batch 4,
and AMP true without entering model construction.

The old benchmark logical record `<removed-content-record>`, canonical checkpoint
`<removed-content-record>`, model/config/loss/graph identities, old A0/A8/E5 artifacts,
and old formal result inventory match their before snapshots.

Ordinary smoke passed 28/28 original entries, 28/28 E5 entries, and 2/2
ST-MGPrompt Full/A8 entries. Restricted real-data smoke completed the
26-model inventory with 24 PASS, 1 OOM, and 1 abnormal process exit.

Exact known-OOM6 preflight produced 4 PASS and 2 OOM, so all26, Full, and A8
exact preflights were intentionally not run. Formal suites and aggregations
remain NOT_RUN.

The execution layer now treats that result as `GTX1060_WINDOWS`-local evidence.
Target machines generate machine-bound, suite-specific Original, E5, Full, and
A8 preflight artifacts. Matching PASS summaries automatically release only the
corresponding formal suite. E5 core is independent of base28/A8 completion;
E5 finalize remains fail-closed until both are complete. No formal run was
started by this engineering revision.
