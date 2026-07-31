# Uniform Batch=4 Result Schema

Every artifact and checkpoint must carry
`base_benchmark_protocol_hash`, `training_batch_profile_id`,
`training_batch_profile_hash`, train/val/test batch sizes,
`gradient_accumulation_steps`, and `effective_train_batch_size`.
Formal trainable runs also carry loss identity, model config/source closure
identity, graph/node/timestamp identity, and matching exact preflight identity.
