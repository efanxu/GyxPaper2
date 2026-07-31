# E2-B implementation report

E2-B integrated TimesNet, MICN, WPMixer, and MultiPatchFormer through the existing benchmark_v2 Registry, NodeSharedAdapter contract, Trainer, Evaluator, Artifact profiles, SDWPF provider, explicit TSLib loader, CPU-only launcher, and independent hardware-preflight/formal workers.

Verification:

- unit tests: 71/71 PASS;
- MICN original 154/58 failure: reproduced and retained;
- MultiPatchFormer original 18/19 failure: reproduced and retained;
- ordinary smoke: final 4/4 PASS, including backward, optimizer step, strict reload, independent evaluation, and artifact validation;
- exact local full-shape: TimesNet/MICN PASS; WPMixer/MultiPatchFormer `FAIL_OOM`; no final `FAIL_NON_OOM`;
- limited real SDWPF: 4/4 PASS with at most 2/1/1 train/validation/evaluation batches;
- formal training/evaluation: NOT_RUN;
- target high-memory hardware preflight: NOT_RUN.

Three early CUDA engineering failures are preserved in `model_smoke_results.json`: TimesNet FP16 cuFFT length 154 and two WPMixer custom-DWT dtype failures. Each was classified `FAIL_NON_OOM`, source-traced, fixed at a precision boundary, and never hidden as OOM.

Final Registry statistics are 28 total, 13 formal-runnable trainable, 2 non-trainable available, and 13 true blocked/unavailable. Of the 13 trainable, 7 are locally full-shape verified and 6 require target-machine preflight. For this stage specifically, TimesNet/MICN are local PASS and WPMixer/MultiPatchFormer require preflight.

No formal output directory was created, no dependency changed, no file was deleted, and E2-C did not start.
