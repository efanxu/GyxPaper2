# E5 batch4 scope27 change: SegRNN and MSGNet

The current E5 evidence scope is
`e5_batch4_scope27_seed2026`: 24 common-loss trainable variants, two
evaluate-only baselines, and one read-only ST-MGPrompt A8 reference.

SegRNN and MSGNet are outside the current denominator. Their status is
`EXCLUDED_FROM_CURRENT_FORMAL_SCOPE`; the reason is
`RESOURCE_REQUIREMENT_EXCEEDS_AVAILABLE_FORMAL_HARDWARE`.

The exclusion changes only current activity:

- they are absent from the scope27 manifest execution entries;
- they are absent from the scope27 run-id map;
- they are absent from readiness and aggregation denominators;
- their existing source, smoke/OOM evidence, logs, failed/incomplete
  directories, and historical artifacts remain untouched.

The old 29-entry E5 files and old uniform-batch4 28/29 files are historical,
read-only evidence. `E5_ACTIVE_SCOPE.json` marks them superseded so they cannot
control the scope27 launcher or resolver.

The scope27 formal path does not require a hardware PASS artifact. This narrow
exception is authorized only when all bindings match:

- scope `e5_batch4_scope27_seed2026`;
- experiment profile `e5_common_loss_architecture_v1`;
- training profile `uniform_train_batch4_v1`;
- one of the 24 active trainable model ids.

All other formal paths retain their prior behavior. Scope, protocol, loss,
model source/config, dataset, run-id, output conflict, artifact completeness,
checkpoint/metrics hash, run status, A8 identity, and 27/27 require-complete
gates remain enabled.
