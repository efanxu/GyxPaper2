#!/usr/bin/env bash
# Machine-gated Original Benchmark MSGNet-only supplement.
#
# The run id and output root are read from UNIFORM_BATCH4_RUN_ID_MAP.json.
# No E5 experiment profile is passed. The training command therefore uses the
# frozen Original Benchmark masked_mse loss and launches as an independent
# Python child process.
#
# Static/no-GPU contract tests:
#   bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_ORIGINAL_MSGNET_LINUX.sh --self-test
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - "${1:-run}" "$PROJECT_ROOT" "$PYTHON_BIN" <<'PYTHON'
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MODE = sys.argv[1]
PROJECT_ROOT = Path(sys.argv[2]).resolve()
PYTHON_BIN = sys.argv[3]
MODEL_ID = "msgnet"
PROFILE_ID = "uniform_train_batch4_v1"
PROFILE_HASH = "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66"
LOSS_ID = "masked_mse"
PREFLIGHT_KIND = "ORIGINAL_MSGNET_ORIGINAL_LOSS_EXACT"
EXPECTED_OUTPUT_ROOT = (
    "custom_models/results/benchmark_v2_uniform_bs4/e2_d_seed2026"
)
DOC_ROOT = (
    PROJECT_ROOT / "custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL"
)
RUN_MAP_PATH = DOC_ROOT / "UNIFORM_BATCH4_RUN_ID_MAP.json"
RAW_ROOT = (
    PROJECT_ROOT
    / "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
)
SUMMARY_ROOT = (
    PROJECT_ROOT
    / "custom_models/results_smoke/benchmark_v2_uniform_bs4"
    / "original_msgnet_supplement"
)
LOG_ROOT = (
    PROJECT_ROOT
    / "custom_models/logs/uniform_bs4/formal/original_msgnet_supplement"
)
MACHINE_FIELDS = (
    "machine_id",
    "hostname",
    "operating_system",
    "python_version",
    "pytorch_version",
    "cuda_version",
    "gpu_name",
    "gpu_uuid",
    "gpu_total_memory",
    "driver_version",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_original_msgnet_row(path: Path) -> dict[str, Any]:
    payload = json_load(path)
    if payload.get("profile_id") != PROFILE_ID:
        raise ValueError("run-id map profile_id mismatch")
    if payload.get("profile_hash") != PROFILE_HASH:
        raise ValueError("run-id map profile_hash mismatch")
    rows = [
        row
        for row in payload.get("runs", [])
        if row.get("model_id") == MODEL_ID
        and row.get("experiment_type") == "original_benchmark"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"run-id map must contain exactly one Original MSGNet row; got {len(rows)}"
        )
    row = dict(rows[0])
    required = {
        "model_id": MODEL_ID,
        "experiment_type": "original_benchmark",
        "batch_profile_id": PROFILE_ID,
        "output_root": EXPECTED_OUTPUT_ROOT,
        "mode": "TRAIN",
        "loss_id": LOSS_ID,
    }
    mismatches = [
        key for key, value in required.items() if row.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"run-id map Original MSGNet row mismatch: {mismatches}")
    run_id = row.get("new_run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
    ):
        raise ValueError("run-id map contains an unsafe or empty new_run_id")
    return row


def raw_pass_blockers(
    raw: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    blockers = [
        f"raw.identity.{key}: mismatch"
        for key, value in expected.items()
        if raw.get(key) != value
    ]
    required = {
        "model_id": MODEL_ID,
        "loss_id": LOSS_ID,
        "training_batch_profile_id": PROFILE_ID,
        "training_batch_profile_hash": PROFILE_HASH,
        "B": 4,
        "T": 144,
        "N": 134,
        "C": 16,
        "H": 10,
        "amp": True,
        "amp_enabled": True,
        "seed": 2026,
        "status": "PASS",
        "forward_completed": True,
        "backward_completed": True,
        "finite_gradients": True,
    }
    blockers.extend(
        f"raw.{key}: mismatch"
        for key, value in required.items()
        if raw.get(key) != value
    )
    if raw.get("prediction_shape") != [4, 134, 10]:
        blockers.append("raw.prediction_shape: expected [4, 134, 10]")
    loss = raw.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
        blockers.append("raw.loss: non-finite")
    if raw.get("experiment_profile_id") is not None:
        blockers.append("raw.experiment_profile_id: E5 PASS is forbidden")
    if raw.get("e5_common_loss_protocol_hash") is not None:
        blockers.append("raw.e5_common_loss_protocol_hash: E5 PASS is forbidden")
    return sorted(set(blockers))


def preflight_blockers(
    *,
    summary: Mapping[str, Any],
    raw: Mapping[str, Any],
    expected: Mapping[str, Any],
    current_machine: Mapping[str, Any],
    current_freeze: Mapping[str, Any],
    raw_path: Path,
    raw_sha256: str,
    freeze_matcher: Callable[
        [Mapping[str, Any], Mapping[str, Any]], tuple[bool, list[str]]
    ] | None = None,
) -> list[str]:
    blockers = raw_pass_blockers(raw, expected)
    required = {
        "artifact_kind": PREFLIGHT_KIND,
        "suite_type": "ORIGINAL_MSGNET_SUPPLEMENT",
        "experiment_type": "original_benchmark",
        "model_id": MODEL_ID,
        "loss_id": LOSS_ID,
        "status": "PASS",
        "training_batch_profile_id": PROFILE_ID,
        "training_batch_profile_hash": PROFILE_HASH,
        "raw_preflight_artifact": str(raw_path),
        "raw_preflight_sha256": raw_sha256,
    }
    blockers.extend(
        f"summary.{key}: mismatch"
        for key, value in required.items()
        if summary.get(key) != value
    )
    bound_machine = summary.get("machine_identity", {})
    blockers.extend(
        f"summary.machine_identity.{key}: mismatch"
        for key in MACHINE_FIELDS
        if bound_machine.get(key) != current_machine.get(key)
    )
    if summary.get("preflight_identity") != expected:
        blockers.append("summary.preflight_identity: mismatch")
    bound_freeze = summary.get("freeze_identity", {})
    if freeze_matcher is None:
        if bound_freeze != current_freeze:
            blockers.append("summary.freeze_identity: mismatch")
    else:
        matched, mismatches = freeze_matcher(bound_freeze, current_freeze)
        blockers.extend(f"summary.freeze_identity.{key}: mismatch" for key in mismatches)
        if not matched and not mismatches:
            blockers.append("summary.freeze_identity: mismatch")
    if summary.get("git_commit") != current_freeze.get("git_commit"):
        blockers.append("summary.git_commit: mismatch")
    checks = summary.get("checks", {})
    for key in (
        "forward",
        "original_formal_loss",
        "backward",
        "finite_loss",
        "finite_gradients",
        "prediction_shape_4_134_10",
    ):
        if checks.get(key) is not True:
            blockers.append(f"summary.checks.{key}: not true")
    return sorted(set(blockers))


def decide_existing_run(
    *, exists: bool, completed_valid: bool, status: str | None, active: bool
) -> str:
    if not exists:
        return "CREATE_NEW"
    if completed_valid:
        return "COMPLETED_VALID_SKIP"
    if status == "RUNNING" and active:
        return "RUNNING_ACTIVE_BLOCK"
    return "STALE_FAILED_INCOMPLETE_BLOCK"


def active_run_pids(run_id: str) -> list[int]:
    active: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return active
    for candidate in proc.iterdir():
        if not candidate.name.isdigit() or int(candidate.name) == os.getpid():
            continue
        try:
            command = (candidate / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if (
            run_id in command
            and "run_benchmark.py" in command
            and (" train " in f" {command} " or "_formal-train-worker" in command)
        ):
            active.append(int(candidate.name))
    return sorted(active)


def completed_valid(
    run_root: Path,
    *,
    validate_run: Callable[..., Mapping[str, Any]],
    protocol_hash: str,
) -> tuple[bool, str]:
    try:
        status = json_load(run_root / "run_status.json")
        if (
            status.get("status") != "COMPLETED"
            or status.get("run_mode") != "formal"
            or status.get("artifact_profile") != "TRAIN"
        ):
            return False, "run_status is not formal TRAIN COMPLETED"
        effective = json_load(run_root / "effective_config.json")
        if effective.get("model_id") != MODEL_ID:
            return False, "effective model_id mismatch"
        if effective.get("loss") != LOSS_ID:
            return False, "effective Original loss mismatch"
        if effective.get("experiment_profile_id") is not None:
            return False, "E5 experiment profile is present"
        if effective.get("training_batch_profile_id") != PROFILE_ID:
            return False, "training profile id mismatch"
        if effective.get("training_batch_profile_hash") != PROFILE_HASH:
            return False, "training profile hash mismatch"
        report = validate_run(
            run_root,
            expected_protocol_hash=protocol_hash,
            expected_training_batch_profile_id=PROFILE_ID,
            expected_training_batch_profile_hash=PROFILE_HASH,
        )
        if report.get("status") != "PASS":
            return False, "artifact validation did not PASS"
    except Exception as exc:
        return False, f"artifact validation failed: {type(exc).__name__}: {exc}"
    return True, "all formal artifacts and frozen identities are valid"


def build_training_command(
    *, python_bin: str, row: Mapping[str, Any], preflight_root: Path
) -> list[str]:
    return [
        python_bin,
        str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
        "train",
        "--model",
        MODEL_ID,
        "--training-profile",
        PROFILE_ID,
        "--output-root",
        str(PROJECT_ROOT / str(row["output_root"])),
        "--run-id",
        str(row["new_run_id"]),
        "--device",
        "cuda",
        "--preflight-root",
        str(preflight_root),
    ]


def run_self_tests() -> int:
    import unittest

    class OriginalMsgnetGateTests(unittest.TestCase):
        def fixture(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
            machine = {
                "machine_id": "linux_gpu_a",
                "hostname": "host-a",
                "operating_system": "Linux 6.x",
                "python_version": "3.11",
                "pytorch_version": "2.x",
                "cuda_version": "12.x",
                "gpu_name": "GPU A",
                "gpu_uuid": "GPU-a",
                "gpu_total_memory": 48 * 1024**3,
                "driver_version": "driver",
            }
            expected = {
                **machine,
                "model_id": MODEL_ID,
                "loss_id": LOSS_ID,
                "training_batch_profile_id": PROFILE_ID,
                "training_batch_profile_hash": PROFILE_HASH,
                "B": 4,
                "T": 144,
                "N": 134,
                "C": 16,
                "H": 10,
                "amp": True,
                "amp_enabled": True,
                "seed": 2026,
            }
            raw = {
                **expected,
                "status": "PASS",
                "forward_completed": True,
                "backward_completed": True,
                "finite_gradients": True,
                "prediction_shape": [4, 134, 10],
                "loss": 1.0,
            }
            freeze = {"git_commit": "commit", "frozen_identity_hash": "freeze"}
            raw_path = Path("/preflight/original-msgnet.json")
            summary = {
                "artifact_kind": PREFLIGHT_KIND,
                "suite_type": "ORIGINAL_MSGNET_SUPPLEMENT",
                "experiment_type": "original_benchmark",
                "model_id": MODEL_ID,
                "loss_id": LOSS_ID,
                "status": "PASS",
                "training_batch_profile_id": PROFILE_ID,
                "training_batch_profile_hash": PROFILE_HASH,
                "machine_identity": machine,
                "git_commit": freeze["git_commit"],
                "freeze_identity": freeze,
                "preflight_identity": expected,
                "raw_preflight_artifact": str(raw_path),
                "raw_preflight_sha256": "raw-sha",
                "checks": {
                    "forward": True,
                    "original_formal_loss": True,
                    "backward": True,
                    "finite_loss": True,
                    "finite_gradients": True,
                    "prediction_shape_4_134_10": True,
                },
            }
            return machine, expected, raw, summary

        def blockers(
            self,
            machine: Mapping[str, Any],
            expected: Mapping[str, Any],
            raw: Mapping[str, Any],
            summary: Mapping[str, Any],
        ) -> list[str]:
            return preflight_blockers(
                summary=summary,
                raw=raw,
                expected=expected,
                current_machine=machine,
                current_freeze={"git_commit": "commit", "frozen_identity_hash": "freeze"},
                raw_path=Path("/preflight/original-msgnet.json"),
                raw_sha256="raw-sha",
            )

        def test_original_msgnet_pass_releases(self) -> None:
            machine, expected, raw, summary = self.fixture()
            self.assertEqual(self.blockers(machine, expected, raw, summary), [])

        def test_e5_msgnet_pass_cannot_release(self) -> None:
            machine, expected, raw, summary = self.fixture()
            raw["loss_id"] = "masked_score_aligned_hybrid"
            raw["experiment_profile_id"] = "e5_common_loss_v1"
            self.assertTrue(self.blockers(machine, expected, raw, summary))

        def test_other_machine_pass_cannot_release(self) -> None:
            machine, expected, raw, summary = self.fixture()
            current = dict(machine)
            current["gpu_uuid"] = "GPU-other"
            self.assertTrue(self.blockers(current, expected, raw, summary))

        def test_oom_and_non_oom_do_not_create_formal_run(self) -> None:
            for failure in ("FAIL_OOM", "FAIL_NON_OOM"):
                with self.subTest(failure=failure):
                    machine, expected, raw, summary = self.fixture()
                    raw["status"] = failure
                    launched = False
                    if not self.blockers(machine, expected, raw, summary):
                        launched = True
                    self.assertFalse(launched)

        def test_existing_directories_are_never_overwritten(self) -> None:
            self.assertEqual(
                decide_existing_run(
                    exists=True,
                    completed_valid=True,
                    status="COMPLETED",
                    active=False,
                ),
                "COMPLETED_VALID_SKIP",
            )
            self.assertEqual(
                decide_existing_run(
                    exists=True,
                    completed_valid=False,
                    status="RUNNING",
                    active=True,
                ),
                "RUNNING_ACTIVE_BLOCK",
            )
            self.assertEqual(
                decide_existing_run(
                    exists=True,
                    completed_valid=False,
                    status="FAILED",
                    active=False,
                ),
                "STALE_FAILED_INCOMPLETE_BLOCK",
            )

        def test_training_command_is_original_and_map_driven(self) -> None:
            row = {
                "output_root": EXPECTED_OUTPUT_ROOT,
                "new_run_id": "value-from-map",
            }
            command = build_training_command(
                python_bin="/env/bin/python", row=row, preflight_root=Path("/preflight")
            )
            self.assertIn("value-from-map", command)
            self.assertNotIn("--experiment-profile", command)
            self.assertEqual(command[0], "/env/bin/python")

        def test_repository_run_map_has_one_frozen_original_row(self) -> None:
            row = load_original_msgnet_row(RUN_MAP_PATH)
            self.assertEqual(row["model_id"], MODEL_ID)
            self.assertEqual(row["loss_id"], LOSS_ID)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OriginalMsgnetGateTests)
    )
    return 0 if result.wasSuccessful() else 1


if MODE == "--self-test":
    raise SystemExit(run_self_tests())
if MODE != "run":
    print(f"ERROR: unsupported argument: {MODE}", file=sys.stderr)
    raise SystemExit(2)
if platform.system() != "Linux":
    print("ERROR: this formal supplement is Linux-only.", file=sys.stderr)
    raise SystemExit(2)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "custom_models/src"))

from benchmark_v2.artifacts import validate_run  # noqa: E402
from benchmark_v2.hardware_preflight import (  # noqa: E402
    machine_identity,
    preflight_artifact_path,
    preflight_identity,
)
from benchmark_v2.protocol import load_protocol  # noqa: E402
from benchmark_v2.training_profiles import load_training_profile  # noqa: E402
from scripts.uniform_batch4_machine_gate import (  # noqa: E402
    current_freeze_identity,
    freeze_identities_match,
)

try:
    row = load_original_msgnet_row(RUN_MAP_PATH)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"FORMAL RUN BLOCKED: {exc}", file=sys.stderr)
    raise SystemExit(4)

profile = load_training_profile(PROFILE_ID)
if profile.profile_hash != PROFILE_HASH:
    print("FORMAL RUN BLOCKED: frozen training profile hash mismatch.", file=sys.stderr)
    raise SystemExit(4)

machine = machine_identity()
freeze = current_freeze_identity()
if not freeze.get("git_commit"):
    print(
        "FORMAL RUN BLOCKED: Git commit unavailable; set UNIFORM_BATCH4_GIT_COMMIT.",
        file=sys.stderr,
    )
    raise SystemExit(4)
expected = preflight_identity(
    MODEL_ID, experiment_profile=None, training_profile=PROFILE_ID
)
raw_path = preflight_artifact_path(
    MODEL_ID,
    root=RAW_ROOT,
    experiment_profile=None,
    training_profile=PROFILE_ID,
)
summary_path = (
    SUMMARY_ROOT / str(machine["machine_id"]) / "ORIGINAL_MSGNET_EXACT_PASS.json"
)
if not raw_path.is_file() or not summary_path.is_file():
    print(
        "FORMAL RUN BLOCKED: current-machine Original MSGNet PASS artifact is missing.",
        file=sys.stderr,
    )
    raise SystemExit(4)
raw = json_load(raw_path)
summary = json_load(summary_path)
blockers = preflight_blockers(
    summary=summary,
    raw=raw,
    expected=expected,
    current_machine=machine,
    current_freeze=freeze,
    raw_path=raw_path,
    raw_sha256=sha256_file(raw_path),
    freeze_matcher=freeze_identities_match,
)
if blockers:
    print("FORMAL RUN BLOCKED: preflight mismatch:", file=sys.stderr)
    for blocker in blockers:
        print(f"- {blocker}", file=sys.stderr)
    raise SystemExit(4)

output_root = PROJECT_ROOT / str(row["output_root"])
run_id = str(row["new_run_id"])
run_root = output_root / run_id
status_value: str | None = None
if run_root.is_dir() and (run_root / "run_status.json").is_file():
    try:
        status_value = str(json_load(run_root / "run_status.json").get("status"))
    except (OSError, json.JSONDecodeError):
        status_value = None
pids = active_run_pids(run_id) if run_root.exists() else []
valid = False
audit_reason = "run directory does not exist"
if run_root.is_dir():
    valid, audit_reason = completed_valid(
        run_root,
        validate_run=validate_run,
        protocol_hash=load_protocol().protocol_hash,
    )
decision = decide_existing_run(
    exists=run_root.exists(),
    completed_valid=valid,
    status=status_value,
    active=bool(pids),
)
print(
    json.dumps(
        {
            "run_id": run_id,
            "run_root": str(run_root),
            "audit_decision": decision,
            "audit_reason": audit_reason,
            "active_pids": pids,
        },
        ensure_ascii=False,
        indent=2,
    )
)
if decision in {"RUNNING_ACTIVE_BLOCK", "STALE_FAILED_INCOMPLETE_BLOCK"}:
    print("FORMAL RUN BLOCKED: existing directory is preserved and will not be touched.", file=sys.stderr)
    raise SystemExit(5)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
log_dir = LOG_ROOT / str(machine["machine_id"]) / f"{stamp}_{os.getpid()}"
log_dir.mkdir(parents=True, exist_ok=False)
manifest_path = log_dir / "machine_execution_manifest.json"
exit_code_path = log_dir / "suite_exit_code.txt"
base_manifest: dict[str, Any] = {
    "schema_version": "uniform_batch4_machine_execution_original_msgnet_v1",
    "suite_type": "ORIGINAL_MSGNET_SUPPLEMENT",
    "experiment_type": "original_benchmark",
    "model_id": MODEL_ID,
    "loss_id": LOSS_ID,
    "run_id": run_id,
    "output_root": str(row["output_root"]),
    "run_directory": str(run_root),
    "machine_identity": machine,
    "git_commit": freeze["git_commit"],
    "training_batch_profile_id": PROFILE_ID,
    "training_batch_profile_hash": PROFILE_HASH,
    "preflight_artifact": str(summary_path),
    "preflight_artifact_sha256": sha256_file(summary_path),
    "raw_preflight_artifact": str(raw_path),
    "raw_preflight_artifact_sha256": sha256_file(raw_path),
    "orchestrator_pid": os.getpid(),
    "training_pid": None,
    "command": None,
    "started_at": utc_now(),
    "finished_at": None,
    "training_exit_code": None,
    "exit_code": None,
    "result_status": "PENDING",
    "audit_decision": decision,
    "audit_reason": audit_reason,
}

if decision == "COMPLETED_VALID_SKIP":
    base_manifest.update(
        {
            "finished_at": utc_now(),
            "training_exit_code": None,
            "exit_code": 0,
            "result_status": "COMPLETED_VALID_SKIPPED",
        }
    )
    write_json(manifest_path, base_manifest)
    exit_code_path.write_text("0\n", encoding="utf-8")
    print(f"COMPLETED_VALID: skipped without touching {run_root}")
    raise SystemExit(0)

command = build_training_command(
    python_bin=PYTHON_BIN, row=row, preflight_root=RAW_ROOT
)
base_manifest["command"] = command
base_manifest["result_status"] = "RUNNING"
train_log = log_dir / "train.log"
try:
    with train_log.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        base_manifest["training_pid"] = process.pid
        write_json(manifest_path, base_manifest)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        training_code = int(process.wait())
except Exception as exc:
    suite_code = 6
    base_manifest.update(
        {
            "finished_at": utc_now(),
            "exit_code": suite_code,
            "result_status": "LAUNCH_FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    )
    write_json(manifest_path, base_manifest)
    exit_code_path.write_text(f"{suite_code}\n", encoding="utf-8")
    print(f"Formal training launch failed: {exc}", file=sys.stderr)
    raise SystemExit(suite_code)

result_status = "FAILED_PRESERVED"
suite_code = training_code
post_reason = "training child returned nonzero"
if training_code == 0:
    post_valid, post_reason = completed_valid(
        run_root,
        validate_run=validate_run,
        protocol_hash=load_protocol().protocol_hash,
    )
    if post_valid:
        result_status = "COMPLETED_VALID"
        suite_code = 0
    else:
        result_status = "FAILED_RESULT_INVALID_PRESERVED"
        suite_code = 7
base_manifest.update(
    {
        "finished_at": utc_now(),
        "training_exit_code": training_code,
        "exit_code": suite_code,
        "result_status": result_status,
        "post_run_audit": post_reason,
    }
)
write_json(manifest_path, base_manifest)
exit_code_path.write_text(f"{suite_code}\n", encoding="utf-8")
raise SystemExit(suite_code)
PYTHON
exit $?
