# E0-B acceptance report

E0-B implements the independent benchmark_v2 runtime only. No external model, graph model, Persistence, MovingAverage or standalone GRU was implemented. The formal registry has 28 unavailable entries. The protocol record is `<removed-content-record>`.

Acceptance is evidenced by the generated `test_results.json`, `framework_smoke_results.json`, and protection record files in this directory. Any `NOT_RUN` item is recorded explicitly in the final manifest; four E0-A TSLib smoke errors remain recorded and intentionally unfixed.

Observed results: 17/17 unit tests PASS; protocol check PASS; ordinary framework smoke PASS and artifact validation PASS; full-shape framework smoke PASS on CUDA with output `(32,134,10)`; real SDWPF data-contract smoke PASS with two windows per split and no training. Formal training runs: none. Canonical and protected-source before/after records match.
