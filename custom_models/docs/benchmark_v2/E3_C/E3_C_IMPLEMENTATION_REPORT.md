# E3-C implementation report

## Outcome

Graph WaveNet, MTGNN, AGCRN, and STID are integrated as native
134-turbine models. All four are `AVAILABLE_TRAINABLE`, support train/evaluate,
passed local exact full-shape forward/loss/backward with AMP, and do not require
a new mandatory target-machine hardware preflight.

Registry after E3-C: 28 total, 26 trainable, 16 locally exact PASS,
10 hardware-preflight-required from earlier stages, 2 available non-trainable,
0 blocked. The explicit TSLib allowlist remains 18.

## Verification

- compileall: PASS
- E3-C focused tests: 13/13 PASS
- complete benchmark_v2 suite: 151/151 PASS in 457.161 seconds
- hardware-preflight launcher regression: 9/9 PASS
- benchmark protocol check: PASS
- graph protocol check: PASS
- ordinary smoke: 4/4 PASS
- exact `(32,144,134,16) -> (32,134,10)` smoke: 4/4 PASS
- limited real SDWPF smoke: 4/4 PASS
- source listing verification: 4/4 PASS

Exact local CUDA peak counters, recorded without inference:

| Model | GPU | Peak allocated | Peak reserved | Status |
|---|---|---:|---:|---|
| Graph WaveNet | GTX 1060 | 6,827,258,368 | 7,669,284,864 | PASS |
| MTGNN | GTX 1060 | 3,305,245,184 | 4,452,253,696 | PASS |
| AGCRN | GTX 1060 | 6,256,342,528 | 6,371,147,776 | PASS |
| STID | GTX 1060 | 133,265,920 | 150,994,944 | PASS |

PyTorch reported GPU total memory as 6,442,319,872 bytes. The Graph WaveNet
allocated/reserved counters are preserved exactly even though they exceed that
reported total; no interpretation or correction was invented.

## Protection

Before/after records match exactly for core ST-MGPrompt, TSLib, the benchmark
protocol/schema/contracts/loss/metrics/engine groups, and the complete graph
protocol. Compatible benchmark extensions are limited to data metadata,
checkpoint/artifact/preflight identity, runtime/CLI wiring, and registry.
Prior E1/E2/E3-A count assertions were updated for the intentional registry
transition. The E3-B directory summary changed because E3-C files and exports
were added under the shared graph-model directory. The historical E3-B
aggregate source listing record is therefore not expected to match because its
shared export/runtime wiring was compatibly extended. The individual
`common.py`, `gcn.py`, `stgcn.py`, and `dcrnn.py` architecture records remain
unchanged, and the full regression suite confirms prior behavior.

No files were deleted or moved, no dependency was changed, and no formal output
root was created.

## Explicitly not run

Formal Full training, formal evaluation, target high-memory preflight, and E4
were not run. The local full-shape smoke is an engineering verification and is
not a formal training/preflight substitute.
