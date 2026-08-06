# Handoff: E5 Batch4 scope27

The current E5 scope is `e5_batch4_scope27_seed2026`: 24 trainable, 2
evaluate-only, and one formal read-only Batch4 A8 reference. Its canonical
output root is
`custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026`.
The old `custom_models/results/benchmark_v2/common_loss_architecture_seed2026`
root is legacy read-only evidence and cannot satisfy current readiness.

The active A8 ID is
`STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference`; the source is the
Batch4 A8 directory under `st_mgprompt_uniform_bs4`, never Batch32. SegRNN,
MSGNet, old scope29 entries, and Transformer retry2 are excluded from the
current denominator and source listing.

Before any cloud run, validate the independent A8 contract, inspect its lock,
freeze-plan, exact preflight, formal run, and readiness first. Only then
validate the active E5 pointer and manifest, inspect the E5 lock, compute the
E5 freeze, review `dry-run`, and run exact GPU preflight for the 24 trainable
entries. Formal `run` requires an exact PASS artifact and never creates one
implicitly. Successful entries receive strict
E5 execution receipts; readiness recomputes file records and validates metrics,
CSV/JSON equality, checkpoints, baseline diagnostics, source/config/precision,
loss, graph, dataset, manifest, run-map, and freeze identities.

Safe resume preserves existing evidence. Exact completed runs are skipped;
missing runs are started; identity-matched incomplete runs are moved through
external-intent archival; identity mismatches and renamed runs require an
explicit single-model quarantine command. Symlinks, active workers, path
traversal, and collisions block the action. E5 never retrains or copies A8;
it consumes the independently trained prerequisite read-only.

Aggregate is allowed only for `COMPLETED_READY_27_OF_27` with
`--require-complete`. Do not use superseded launchers, old manifests, legacy
roots, or automatic quarantine. The active launcher has no shutdown side
effect; any legacy autoshutdown wrapper remains outside the formal command
sequence.
