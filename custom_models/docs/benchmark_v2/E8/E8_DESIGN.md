# E8 Prompt and Cross-Fusion Design

## Scope and evidence roles

E8 has exactly eleven evidence slots. Internal causal evidence is A0/A4/A5/A6/A7. External context is exactly PatchTST, iTransformer, TimeXer, MultiPatchFormer, TimeMixer, and TimeFilter from original26. E5/common-loss, A8, smoke, batch32-as-batch4 relabeling, and excluded models are forbidden.

The external table is native-training-system architecture/end-to-end context. It cannot be interpreted as a unified-loss structure-controlled comparison.

## Frozen internal semantics

- A4 changes only `macro_prompt_pooling`, from attention pooling to equal-weight mean pooling; `macro_prompt_len=4` remains unchanged.
- A5 changes only `cross_fusion_recent_len`, from 24 to 6; both Reverse Cross directions remain enabled.
- A6 changes only `macro_to_fine_exclude_recent_len`, from 0 to 24, so Macro-to-Fine attention uses the early-history query window while retaining both directions.
- A7 changes only `share_cross_attention_projections`, from `false` to `true`, so both Cross Fusion directions share the same attention projection module.

A0 versus A6 isolates the Macro-to-Fine query-window change, while A0 versus A7 isolates projection sharing. A5 answers a different question by shortening only the reverse context. E8 internal causal evidence remains pending until the A4–A7 formal runs satisfy the Batch4 protocol.

## Real tensor and mechanism contract

VADSP produces `x_fine` and `x_coarse`; both preserve `[B,L,N,D]`. With current formal input dimensions the shape is `[4,144,134,64]`. Graph-temporal encoders produce `h_fine/h_coarse`. Cross Fusion produces updated Fine/Coarse histories; there is no single canonical `post_fusion` tensor, so that semantic is `NOT_APPLICABLE`. Decoder Fine/Coarse inputs and prediction-head representations are captured at their real modules.

Cross Fusion is two `MultiheadAttention` paths plus a learned sigmoid gate, residual addition, and LayerNorm. Real attention/gate statistics may be exported. If a future implementation lacks these objects, the fields become `NOT_APPLICABLE`.

Macro Prompt is a latent `[B,N,4,D]` projection of attention-pooled graph-enhanced Coarse history. It has no physical trend scalar/logit/head, so sign accuracy, physical trend correlation, and confusion matrix are `NOT_APPLICABLE`.

ST Prompt remains horizon-conditioned with node, future-step, and granularity embeddings in all current A0–A8 variants. The former Temporal-Granularity Prompt definition and Node-Temporal Prompt A9 are not part of the formal component-ablation set.

## Read-only diagnostics

Hooks are disabled by default and exist only inside a context manager. Formal diagnostics require five valid current-definition internal artifacts, use `eval()` plus inference mode, and must preserve predictions, parameters, buffers, checkpoint identity, RNG state, and the caller's train/eval mode. Exports go only under the E8 analysis root.

Current internal artifacts are train batch32 and therefore are not eligible for formal E8 evaluate-only diagnostics.

## Readiness and fail-closed behavior

`CORE_E8_READY=11/11` requires internal 5/5 and external 6/6. Representation, Cross Fusion, Macro Prompt, ST Prompt, paired-window, and grouped analyses have separate readiness fields. `--require-complete` exits nonzero before formal CSV/Markdown/Excel creation whenever core readiness is incomplete.

Grouped analysis additionally requires `E10_GROUP_DEFINITION_MANIFEST.json`, `test_window_identity_manifest.json`, and `shared_difficult_top10_window_ids.json`. No thresholds are invented.
