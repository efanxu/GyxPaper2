# Uniform Batch=4 Aggregation Policy

E4 may read only batch4 original benchmark artifacts and batch4 Full. E5 may
read only batch4 common-loss artifacts and the new batch4 A8 reference.
Readiness rejects any id/hash/batch/loss/config/source mismatch as
`BLOCKED_MIXED_BATCH_PROFILE`; aggregation must not emit a final table.
