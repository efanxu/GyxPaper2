# E5 config-diff policy

Only experiment profile, structured loss identity, run-id, output-root,
experiment-config hash, preflight identity, and provenance may differ.
Architecture, source closure, model config, data, optimizer, scheduler, batch,
seed, AMP, checkpoint selection, graph, node order, teacher forcing and
timestamp policy are frozen. Any other difference is `BLOCKED_CONFIG_DIFF`.
