# HANDOFF E2-B

E2-B is complete and stopped before formal runs and E2-C.

Integrated models and sources:

- TimesNet — `Time-Series-Library/models/TimesNet.py`, content record `<removed-content-record>`; final `AVAILABLE_TRAINABLE`.
- MICN — `Time-Series-Library/models/MICN.py`, content record `<removed-content-record>`; final `AVAILABLE_TRAINABLE`.
- WPMixer — `Time-Series-Library/models/WPMixer.py`, content record `<removed-content-record>`; final `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`.
- MultiPatchFormer — `Time-Series-Library/models/MultiPatchFormer.py`, content record `<removed-content-record>`; final `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`.

The explicit TSLib allowlist has exactly 12 IDs: `dlinear, lightts, tide, segrnn, transformer, patchtst, itransformer, timexer, timesnet, micn, wpmixer, multipatchformer`.

All four use node-shared `(B,T,N,16)->(B*N,T,16)->(B*N,10,16)->(B,N,10)`, dynamic verified selection of `Patv_clean_for_input`, and normalized `Patv_raw` supervision. No future target, mask, observed covariate, weather, calendar mark, graph, node embedding, or node-specific head enters forward.

Final configurations:

- TimesNet: `d_model=32,d_ff=32,e_layers=2,top_k=5,num_kernels=6,c_out=16`; FFT FP32 only, remaining learned path AMP. Upstream batch-mean FFT is isolated by node groups.
- MICN: `d_model=32,d_layers=1,conv=[12,16],decomp=[13,17],isometric=[13,10],c_out=16`; decoder and zero marks both length 154.
- WPMixer: `d_model=256,db2,level=1,patch=16,stride=8,tfactor=5,dfactor=5`; DWT/IDWT FP32, learned mixers AMP.
- MultiPatchFormer: `d_model=256,d_ff=512,e_layers=1,n_heads=8`; patches `[8,16,24,32]`, strides `[8,8,7,7]`, paddings `[0,8,0,8]`, all 18 tokens.

MICN historical error was reproduced exactly: seasonal decoder length `144+10=154` versus generic `48+10=58` marks at `MICN.py:167 -> Embed.py:124`. The Adapter supplies observed history plus ten zero decoder steps and 154 zero marks. It does not change lookback/horizon or TSLib.

MultiPatchFormer historical error was reproduced exactly: source branches are `18,18,19,20` tokens and fail at branch 3 with 18/19. Final geometry is `18,18,18,18`; no silent crop or branch disablement. A second horizon-10 integer-division head defect was repaired with exact cumulative forecast-chunk widths. All four patch embedding branches receive gradients.

WPMixer retains both db2 approximation/detail bands and both ResolutionBranches. Relative to Fixed Dual, both WPMixer and MultiPatchFormer are fixed multiresolution external comparisons, but neither uses VADSP, dynamic gate, Graph, Prompt, or DWU. No smoke/test/OOM result selected capacity.

Verification:

- tests: 71/71 PASS;
- ordinary smoke: 4/4 final PASS; early TimesNet/WPMixer `FAIL_NON_OOM` attempts are retained;
- exact local full-shape: TimesNet PASS peak 5,414,120,960 B; MICN PASS peak 1,956,011,520 B; WPMixer `FAIL_OOM` peak record 12,459,016,192 B; MultiPatchFormer `FAIL_OOM` peak record 12,657,478,144 B;
- limited real SDWPF: 4/4 PASS, maximum 2/1/1 batches;
- strict reload/artifact validation: PASS for every final ordinary and real run;
- target high-memory preflight: NOT_RUN;
- formal training/evaluation: NOT_RUN.

Registry: 28 total; 13 formal-runnable trainable; 7 locally full-shape verified; 6 hardware-preflight required; 2 available non-trainable; 13 true blocked/unavailable.

Protection: protocol remains `<removed-content-record>`; Canonical remains `<removed-content-record>`. Review the six matching before/after snapshot pairs. TSLib, ST-MGPrompt, E1-A, E1-B, E2-A, P0–P5/A0–A8, dependencies, and formal results were not modified.

Runbook: `custom_models/docs/benchmark_v2/E2_B/E2_B_RUNBOOK.md`.

Important traps: do not flatten turbines into TimesNet's shared FFT batch; do not restore MICN's 58-step marks; do not crop MultiPatchFormer tokens; do not enable WPMixer's internal DWT autocast; do not reuse mismatched preflight; do not lower WPMixer/MultiPatchFormer capacity after OOM.

The only next stage is E2-C: TimeMixer, TSMixer, FreTS. Reuse the explicit loader, NodeSharedAdapter contract, dynamic single-target output, history-only/future rejection rules, independent hardware preflight, and unified Trainer/Evaluator/Artifact system. E2-C was not started.
