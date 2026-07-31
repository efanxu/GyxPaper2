#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON:-python}
PROFILE="uniform_train_batch4_v1"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"

failed=0
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "gru" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "dlinear" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "lightts" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "tide" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "segrnn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "transformer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "patchtst" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "itransformer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "timexer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "timesnet" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "micn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "wpmixer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "multipatchformer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "timemixer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "tsmixer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "frets" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "crossformer" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "msgnet" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "timefilter" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "gcn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "stgcn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "dcrnn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "graph_wavenet" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "mtgnn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "agcrn" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "stid" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1
"$PYTHON_BIN" scripts/uniform_batch4_generate.py collect-preflight --suite "all26"
exit "$failed"
