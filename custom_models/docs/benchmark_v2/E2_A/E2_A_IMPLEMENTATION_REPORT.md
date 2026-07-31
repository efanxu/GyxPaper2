# E2-A implementation report

E2-A repaired E1-B hardware release semantics and integrated Transformer, PatchTST, iTransformer, and TimeXer through the existing Registry, Trainer, Evaluator, Artifact system, `NodeSharedAdapter`, SDWPF provider, and explicit TSLib loader.

TiDE and SegRNN migrated from `BLOCKED_FULL_SHAPE_OOM` to `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`. Both now have `supports_train=true`, `supports_evaluate=true`, `formal_hardware_preflight_required=true`, `local_full_shape_status=FAIL_OOM`, and `non_oom_engineering_checks=PASS`. Original OOM summary, peak record, local GTX 1060 identity, and E1-B artifact path remain in Registry and the original E1-B JSON remains unchanged.

The train entry is now a CPU-only parent launcher. Models requiring hardware release automatically run an exact independent preflight worker; a matching PASS automatically continues to a separate formal worker within the same user request. DLinear, LightTS, iTransformer, and TimeXer launch a fresh formal worker directly. No long-lived process trains multiple models.

Verification:

- unit tests: 60/60 PASS;
- ordinary E2-A smoke: 4/4 PASS;
- exact full-shape: iTransformer/TimeXer PASS; Transformer/PatchTST FAIL_OOM; no FAIL_NON_OOM;
- limited real data: 4/4 PASS, at most 2/1/1 train/validation/evaluation batches;
- strict reload and artifact validation: PASS for all ordinary/real runs;
- formal training/evaluation: NOT_RUN.

The formal result root was not created. All E2-A outputs are under `custom_models/results_smoke/benchmark_v2/e2_a`.
