# E2-D implementation report

E2-D integrated Crossformer, MSGNet, and TimeFilter through the existing
explicit loader, Registry, node-shared adapter contract, Trainer, Evaluator,
checkpoint/artifact system, SDWPF provider, and CPU-only preflight launcher.

Final engineering outcome:

- Crossformer: ordinary/real/exact full-shape PASS; local peak allocated
  5,635,071,488 B; `AVAILABLE_TRAINABLE`.
- MSGNet: ordinary and final CPU-limited real smoke PASS; exact local
  `FAIL_OOM` in forward at 13,574,127,616 B peak allocated record;
  `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`.
- TimeFilter: ordinary and CPU-limited real smoke PASS; exact local `FAIL_OOM`
  in forward at 12,233,284,096 B peak allocated record;
  `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`.

The first MSGNet real-data CUDA attempt failed OOM and a CPU attempt with
default thread scheduling was interrupted; both directories are retained.
Neither result selected capacity. The final bounded CPU 1-thread real run
preserved all 134 nodes and the same adapter/model semantics.

Registry remains 28 entries: 19 trainable (9 locally exact PASS and 10
preflight-required), 2 non-trainable available, and 7 blocked/unavailable.
The TSLib allowlist is exactly 18 IDs.

Formal training, formal evaluation, target high-memory preflight, and E3 were
not started. The formal E2-D output root was not created.
