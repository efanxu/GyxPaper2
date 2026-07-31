#!/usr/bin/env bash
set +e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PROJECT_ROOT
bash "$SCRIPT_DIR/benchmark_v2_run_all_28_linux.sh"
code=$?
sync
/usr/bin/shutdown -h now
exit "$code"
