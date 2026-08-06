# E5 common-loss protocol

- Profile: `e5_common_loss_architecture_v1` (CLI: `e5_common_loss_v1`)
- Base benchmark protocol: `<removed-content-record>`
- E5 protocol record: `<removed-content-record>`
- Loss: `masked_score_aligned_hybrid`
- Loss profile record: `<removed-content-record>`
- Seed: `2026`
- Active Batch4 entries: 24 trainable + 2 evaluate-only + 1 independent A8
  prerequisite reference = 27.
- The current root is `custom_models/results/benchmark_v2_uniform_bs4/`;
  the old scope29 root is read-only historical evidence.

The global benchmark default remains `masked_mse`. E5 is an explicit overlay.
Checkpoint selection remains validation official Score H10, lower is better.
