# Uniform Batch=4 Design

`uniform_train_batch4_v1` is the only new training batch profile. It overlays the frozen
Benchmark Protocol v1 (`<removed-content-record>`) without changing that file
or its logical record. Train/validation/test batches are 4/4/4, AMP is enabled,
gradient accumulation is 1, and effective train batch is 4.

The overlay changes DataLoader batch formation, artifact identity, run-id,
output namespace, and preflight identity only. It does not enter model
construction. Batch fallback, per-model batches, activation checkpointing,
offload, model parallelism, pruning, quantization, and capacity changes are
forbidden. Batch=4 creates more optimizer updates per epoch than batch=32.
