# Uniform Batch=4 Result Schema

Every artifact and checkpoint must carry
`base_benchmark_protocol_record`, `training_batch_profile_id`,
`training_batch_profile_record`, train/val/test batch sizes,
`gradient_accumulation_steps`, and `effective_train_batch_size`.
Formal trainable runs also carry loss identity, model config/source listing
identity, graph/node/timestamp identity, and matching exact preflight identity.
