# E2-C implementation report

E2-C integrated TimeMixer, TSMixer, and FreTS through the existing explicit
TSLib loader, Registry, node-shared Adapter contract, unified Trainer,
Evaluator, checkpoint, SDWPF provider, artifact schemas, CPU-only launcher, and
independent preflight/formal workers.

Verification outcome:

- focused E2-C regression: 11/11 PASS;
- hardware-preflight dummy/process-runner regression: 9/9 PASS;
- full benchmark_v2 unittest discovery: 84/84 PASS;
- compileall: PASS;
- protocol-check: PASS, hash
  `0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`;
- ordinary smoke: 3/3 PASS;
- limited real SDWPF smoke: 3/3 PASS;
- exact local full-shape: TSMixer PASS; TimeMixer/FreTS `FAIL_OOM`;
- no final `FAIL_NON_OOM`.

The final Registry contains 28 entries: 16 formal-runnable trainable (8 local
exact PASS and 8 exact-preflight-required), 2 available non-trainable, and 10
true blocked/unavailable.

FreTS type diagnostics confirmed the local source string/int mismatch and
froze string `"0"`. Raw CUDA AMP passed, so final execution keeps native AMP;
the earlier preventive FP32 attempt is preserved as superseded evidence.

No formal output root was created. No formal training, formal evaluation,
target high-memory preflight, dependency installation, source download, file
deletion, or E2-D work occurred.

Two prompt-listed prerequisite files, `实验总Plan.md` and `HANDOFF.md`, were
absent from the entire workspace. The available Protocol, Canonical identity,
E2-A policy, E2-B handoff, and benchmark_v2 contracts were used without
inventing their contents.
