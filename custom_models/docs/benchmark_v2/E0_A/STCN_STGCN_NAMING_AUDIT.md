# STCN / STGCN Naming Audit

## Conclusion: Situation C — evidence insufficient / implementation absent

No trusted `STCN.py`, `STGCN.py`, class, forward path, adjacency provider, temporal graph block, configuration, README citation, or old result metadata was found in the local UTF-8 search scope (`custom_models`, `Time-Series-Library`, `configs`, `scripts`, `tools`, `tests`, `docs`). `Time-Series-Library/run.py:143-149` has only generic GCN-related argparse flags; it does not construct a GCN/STCN/STGCN model. The `models` directory has no GCN, STCN, STGCN, DCRNN, Graph WaveNet, MTGNN, AGCRN or STID source.

Because there is no model class or forward to inspect, the audit cannot prove whether `STCN` means classical STGCN or an independent STCN. Therefore this row is `NAME_CONFLICT`, not a guessed rename.

```text
status = NAME_CONFLICT
canonical_name = STCN_STGCN
recommended_canonical_name = UNKNOWN
legacy_alias = UNKNOWN
STGCN status = MISSING
```

## Required E3-A follow-up

Locate or deliberately select a source with paper identity, license/notice, temporal convolution + graph convolution block definition, adjacency normalization/direction, and forward shape. Record whether it is classical STGCN (then canonicalize to `STGCN`, preserving `STCN` as a legacy alias) or a distinct STCN (then keep `STCN` and add STGCN separately only if source evidence exists). Do not modify names, registry keys, filenames or old results in E0-A or before evidence is complete.
