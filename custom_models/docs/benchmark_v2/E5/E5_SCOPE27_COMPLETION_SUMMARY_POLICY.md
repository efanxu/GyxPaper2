# E5 scope27 completion summary policy

The live completion summary is
`logs/benchmark_v2/e5_batch4_scope27/e5_scope27_final_status.env`.
It records run-all status, failure count, readiness status and exit code,
aggregate status and exit code, final exit code, and completion time.

The live evidence manifest is
`logs/benchmark_v2/e5_batch4_scope27/e5_scope27_evidence_manifest.json`.
It contains exactly 27 entries and records checkpoint, metrics, effective
config, run status, artifact manifest, and A8 protocol-evidence hashes where
applicable.

Neither file can report complete unless the scope27 gate independently resolves
24 trainable formal runs, two evaluate-only formal runs, and the read-only A8
reference as ready. Historical and smoke artifacts are not eligible.
