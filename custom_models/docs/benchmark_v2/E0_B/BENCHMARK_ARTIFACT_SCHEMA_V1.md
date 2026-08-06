# Artifact schema v1

Profiles are `TRAIN`, `EVALUATE_ONLY`, `NON_TRAINABLE`, `REFERENCE_ONLY`, `SMOKE`, and `FAILED`. Required files are declared in `benchmark_v2.artifacts.PROFILES`; missing files are errors, never silent completion.

JSON and CSV use temp-file, flush/fsync, `os.replace` writes. Run paths are root-confined, duplicate runs are rejected, and resume checkpoints verify model ID, protocol record, resolved config record and effective config record. A formal discoverable run must be under `custom_models/results/benchmark_v2`, be `formal` and `COMPLETED`, pass schema validation, match the protocol record, belong to the explicit registry, and contain H3/H6/H10 metrics.

