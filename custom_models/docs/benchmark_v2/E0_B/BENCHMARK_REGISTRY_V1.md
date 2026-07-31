# benchmark registry v1

The registry is `custom_models/src/benchmark_v2/registry/benchmark_registry.json` and is loaded explicitly. It does not scan `models/`, import TSLib, import ST-MGPrompt, download code, or create dummy models.

The 28 canonical IDs are:

`persistence`, `moving_average`, `gru`, `dlinear`, `lightts`, `tide`, `segrnn`, `transformer`, `patchtst`, `itransformer`, `timexer`, `timesnet`, `micn`, `wpmixer`, `multipatchformer`, `timemixer`, `tsmixer`, `frets`, `crossformer`, `msgnet`, `timefilter`, `gcn`, `stcn_stgcn_unresolved`, `dcrnn`, `graph_wavenet`, `mtgnn`, `agcrn`, `stid`.

All entries are unavailable in E0-B (`factory=null`, `supports_train=false`, `supports_evaluate=false`). `stcn_stgcn_unresolved` is `NAME_CONFLICT`, `BLOCKED_PENDING_E3_A`, with no STCN/STGCN aliases and no recommended canonical name.

