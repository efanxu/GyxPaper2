# E2-A TSLib source audit

E2-A uses only the local `THUML/Time-Series-Library` checkout through the explicit lazy loader. The allowlist is exactly eight IDs: `dlinear`, `lightts`, `tide`, `segrnn`, `transformer`, `patchtst`, `itransformer`, `timexer`. No directory scanning, network access, TSLib trainer, TSLib dataset, or automatic model registration is used.

| model | exact source | SHA256 | observed forecast output |
|---|---|---|---|
| Transformer | `Time-Series-Library/models/Transformer.py` | `46ded6c516fd03951bce0a82c3f4245f0247efd01fb5bde775145cfbb6d8435d` | `(B*N,10,1)` with `c_out=1` |
| PatchTST | `Time-Series-Library/models/PatchTST.py` | `29835d4fedddd3cbee26cc60d6a54c04513f5a9ca32e4a0d08de2c9e059084e5` | `(B*N,10,16)` |
| iTransformer | `Time-Series-Library/models/iTransformer.py` | `7fdc721d041b0f8f63be8fa794ecd68422fd958c7c8d449026320fd9f368788e` | `(B*N,10,16)` |
| TimeXer | `Time-Series-Library/models/TimeXer.py` | `b334d7869544d0a5de7a35501d0342d7bd0be4ebda06ccc045855a42819728c0` | `(B*N,10,1)` on the `MS` path |

`registry-list`, `registry-show`, and `protocol-check` do not import any TSLib model. `create_model` resolves one allowlisted file, verifies its provenance hash, loads it lazily, and restores `sys.path`. Protected before/after TSLib snapshots cover `models`, `layers`, `exp`, `data_provider`, `run.py`, and `LICENSE`.

No TSLib file was modified.
