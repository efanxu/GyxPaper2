# E5 config-diff policy

Only experiment profile, structured loss identity, run-id, output-root,
experiment-config record, preflight identity, and provenance may differ.
E5 may additionally record its uniform FP32 override and the PatchTST encoder
activation-checkpointing flag. Activation checkpointing changes only retained
activations and recomputation; it does not change parameters, optimizer,
batch, or architecture. Source listing, model config, data, optimizer,
scheduler, batch, seed, checkpoint selection, graph, node order, teacher
forcing and timestamp policy are otherwise frozen. Any other difference is
`BLOCKED_CONFIG_DIFF`.
