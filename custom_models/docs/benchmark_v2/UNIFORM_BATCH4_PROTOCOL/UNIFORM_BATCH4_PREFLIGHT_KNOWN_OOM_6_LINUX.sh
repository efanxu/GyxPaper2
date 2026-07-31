#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON:-python}
PROFILE="uniform_train_batch4_v1"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"

failed=0
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "segrnn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "transformer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "patchtst" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "frets" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "msgnet" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "timefilter" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" scripts/uniform_batch4_generate.py collect-preflight --suite "known_oom6"
exit "$failed"
