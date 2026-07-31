#!/usr/bin/env bash
# Run the machine-gated Original MSGNet supplement and always shut Linux down
# after the suite command returns. --self-test never shuts the machine down.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SUITE_COMMAND="$PROJECT_ROOT/custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_ORIGINAL_MSGNET_LINUX.sh"

if [[ "${1:-}" == "--self-test" ]]; then
  bash "$SUITE_COMMAND" --self-test
  exit $?
fi

LOG_ROOT="$PROJECT_ROOT/custom_models/logs/uniform_bs4/formal/original_msgnet_supplement/autoshutdown"
mkdir -p "$LOG_ROOT"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
EXIT_CODE_LOG="$LOG_ROOT/suite_exit_code_${stamp}_$$.txt"

set +e
bash "$SUITE_COMMAND"
code=$?
printf '%s\n' "$code" >"$EXIT_CODE_LOG"
sync
/usr/bin/shutdown -h now
exit $code
