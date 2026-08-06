# E3-A graph identity and preflight policy

Every future graph-model resolved/effective config, model summary, protocol check, run status, prediction metadata, checkpoint metadata, and hardware-preflight artifact must contain:

`graph_id`, `graph_protocol_record`, `node_order_record`, `graph_bundle_record`, `location_source_record`, `selected_k`, and `graph_support_names`.

Strict checkpoint reload and evaluation must reject any mismatch. A graph preflight PASS is reusable only when the existing model/config/protocol/source/B/T/N/C/H/AMP/forward/backward/status identity also exactly matches all three graph identity records.

E3-A implements this as immutable identity metadata plus fail-closed validation and dummy-only regression. It does not run a graph preflight. Existing non-graph artifacts omit graph fields and their config/checkpoint/preflight identities remain unchanged.

Different graph content, node order, or graph protocol cannot share a checkpoint or preflight PASS.
