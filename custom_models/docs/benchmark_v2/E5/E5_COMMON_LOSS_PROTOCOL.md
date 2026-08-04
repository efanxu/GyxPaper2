# E5 common-loss protocol

- Profile: `e5_common_loss_architecture_v1` (CLI: `e5_common_loss_v1`)
- Base benchmark protocol: `0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`
- E5 protocol hash: `a7c5b5a5b15b5ab70d08bf926b9de351f05740940f16409c801eed419287141a`
- Loss: `masked_score_aligned_hybrid`
- Loss profile hash: `0fc1fca238d7161d3a0257d639a11df8cc92ad2a1c4d61eab234338fa24b59fa`
- Seed: `2026`
- Active Batch4 entries: 24 trainable + 2 evaluate-only + 1 independent A8
  prerequisite reference = 27.
- The current root is `custom_models/results/benchmark_v2_uniform_bs4/`;
  the old scope29 root is read-only historical evidence.

The global benchmark default remains `masked_mse`. E5 is an explicit overlay.
Checkpoint selection remains validation official Score H10, lower is better.
