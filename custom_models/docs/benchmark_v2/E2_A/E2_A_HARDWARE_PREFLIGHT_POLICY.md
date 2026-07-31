# E2-A hardware preflight policy

`AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` means all non-OOM engineering checks pass but an exact local full-shape attempt failed only because of CUDA memory. It is trainable after a matching PASS; it is neither a full-shape PASS nor a permanent block.

The parent launcher does not import/build a model or hold CUDA tensors. It checks a PASS artifact and, when absent, starts:

1. an independent preflight Python worker using `Path(sys.executable).resolve()`;
2. exact `B=32,T=144,N=134,C=16,H=10`, AMP, forward, masked loss, backward;
3. worker exit, releasing its complete CUDA context;
4. a new formal-training Python worker, only after a matching PASS.

PASS matching requires identical `model_id`, `model_config_hash`, `protocol_hash`, `source_hash`, `B/T/N/C/H`, AMP, completed forward/backward, and `status=PASS`.

`FAIL_OOM` writes `hardware_preflight.json`, returns nonzero, creates no formal run, and never changes capacity. `FAIL_NON_OOM` additionally writes `engineering_block.json` with `runtime_status=BLOCKED_NON_OOM`; it also returns nonzero and starts no training.

Default artifact pattern:

`custom_models/results_smoke/benchmark_v2/hardware_preflight/<model>/<identity-hash>/hardware_preflight.json`.

Current exact identities include:

- TiDE: `tide/8ce0530a7194465a8064/hardware_preflight.json`
- SegRNN: `segrnn/74380f45d8773b6f9509/hardware_preflight.json`
- Transformer: `transformer/76ea8eedcd747b1e3c67/hardware_preflight.json`
- PatchTST: `patchtst/88c28fc94eac08f63f42/hardware_preflight.json`

RTX 4060 hardware preflight was not run in E2-A.
