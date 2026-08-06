# E2-A TSLib source audit

E2-A uses only the local `THUML/Time-Series-Library` checkout through the explicit lazy loader. The allowlist is exactly eight IDs: `dlinear`, `lightts`, `tide`, `segrnn`, `transformer`, `patchtst`, `itransformer`, `timexer`. No directory scanning, network access, TSLib trainer, TSLib dataset, or automatic model registration is used.

| model | exact source | content record | observed forecast output |
|---|---|---|---|
| Transformer | `Time-Series-Library/models/Transformer.py` | `<removed-content-record>` | `(B*N,10,1)` with `c_out=1` |
| PatchTST | `Time-Series-Library/models/PatchTST.py` | `<removed-content-record>` | `(B*N,10,16)` |
| iTransformer | `Time-Series-Library/models/iTransformer.py` | `<removed-content-record>` | `(B*N,10,16)` |
| TimeXer | `Time-Series-Library/models/TimeXer.py` | `<removed-content-record>` | `(B*N,10,1)` on the `MS` path |

`registry-list`, `registry-show`, and `protocol-check` do not import any TSLib model. `create_model` resolves one allowlisted file, verifies its provenance record, loads it lazily, and restores `sys.path`. Protected before/after TSLib snapshots cover `models`, `layers`, `exp`, `data_provider`, `run.py`, and `LICENSE`.

No TSLib file was modified.
