#!/usr/bin/env bash
# Exact, machine-bound preflight for the Original Benchmark MSGNet supplement.
#
# Usage:
#   export PYTHON=/absolute/path/to/env_tslib/bin/python
#   export UNIFORM_BATCH4_GIT_COMMIT=<frozen-commit>  # required for archives without .git
#   bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_PREFLIGHT_ORIGINAL_MSGNET_LINUX.sh
#
# This script never consumes or overwrites a preflight from another machine.
# In particular, historical Windows/GTX1060 FAIL_OOM evidence and E5 artifacts
# have different identities and paths and cannot block or release this preflight.
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
from typing import Any, Mapping


MODE = sys.argv[1]
PROJECT_ROOT = Path(sys.argv[2]).resolve()
PYTHON_BIN = sys.argv[3]
PROFILE_ID = "uniform_train_batch4_v1"
PROFILE_HASH = "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66"
MODEL_ID = "msgnet"
LOSS_ID = "masked_mse"
PREFLIGHT_KIND = "ORIGINAL_MSGNET_ORIGINAL_LOSS_EXACT"
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
    / "custom_models/logs/uniform_bs4/preflight/original_msgnet_supplement"
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


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def raw_pass_blockers(
    raw: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    blockers = [
        f"identity.{key}: mismatch"
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
        f"{key}: expected {value!r}, got {raw.get(key)!r}"
        for key, value in required.items()
        if raw.get(key) != value
    )
    if raw.get("prediction_shape") != [4, 134, 10]:
        blockers.append("prediction_shape: expected [4, 134, 10]")
    loss = raw.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
        blockers.append("loss: expected a finite Original formal loss")
    if raw.get("experiment_profile_id") is not None:
        blockers.append("experiment_profile_id: E5 artifact cannot release Original MSGNet")
    if raw.get("e5_common_loss_protocol_hash") is not None:
        blockers.append("e5_common_loss_protocol_hash: E5 artifact cannot release Original MSGNet")
    for key in (
        "model_config_hash",
        "model_source_closure_hash",
        "graph_protocol_hash",
        "graph_bundle_hash",
        "node_order_hash",
    ):
        if not raw.get(key):
            blockers.append(f"{key}: missing")
    return sorted(set(blockers))


def summary_blockers(
    summary: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    machine: Mapping[str, Any],
    freeze: Mapping[str, Any],
    raw_path: Path,
    raw_sha256: str,
) -> list[str]:
    blockers: list[str] = []
    required = {
        "artifact_kind": PREFLIGHT_KIND,
        "suite_type": "ORIGINAL_MSGNET_SUPPLEMENT",
        "experiment_type": "original_benchmark",
        "model_id": MODEL_ID,
        "loss_id": LOSS_ID,
        "status": "PASS",
        "training_batch_profile_id": PROFILE_ID,
        "training_batch_profile_hash": PROFILE_HASH,
        "git_commit": freeze.get("git_commit"),
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
        if bound_machine.get(key) != machine.get(key)
    )
    if summary.get("preflight_identity") != expected:
        blockers.append("summary.preflight_identity: mismatch")
    bound_freeze = summary.get("freeze_identity", {})
    if (
        not freeze.get("frozen_identity_hash")
        or bound_freeze.get("frozen_identity_hash")
        != freeze.get("frozen_identity_hash")
    ):
        blockers.append("summary.freeze_identity: mismatch")
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


def run_self_tests() -> int:
    import unittest

    class PreflightContractTests(unittest.TestCase):
        def expected(self) -> dict[str, Any]:
            return {
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
                "model_id": MODEL_ID,
                "loss_id": LOSS_ID,
                "training_batch_profile_id": PROFILE_ID,
                "training_batch_profile_hash": PROFILE_HASH,
                "model_config_hash": "config",
                "model_source_closure_hash": "source",
                "graph_protocol_hash": "graph",
                "graph_bundle_hash": "bundle",
                "node_order_hash": "nodes",
                "B": 4,
                "T": 144,
                "N": 134,
                "C": 16,
                "H": 10,
                "amp": True,
                "amp_enabled": True,
                "seed": 2026,
            }

        def passing(self) -> dict[str, Any]:
            return {
                **self.expected(),
                "status": "PASS",
                "forward_completed": True,
                "backward_completed": True,
                "finite_gradients": True,
                "prediction_shape": [4, 134, 10],
                "loss": 1.0,
            }

        def test_original_pass_is_accepted(self) -> None:
            self.assertEqual(raw_pass_blockers(self.passing(), self.expected()), [])

        def test_e5_pass_is_rejected(self) -> None:
            raw = self.passing()
            raw["loss_id"] = "masked_score_aligned_hybrid"
            raw["experiment_profile_id"] = "e5_common_loss_v1"
            self.assertTrue(raw_pass_blockers(raw, self.expected()))

        def test_oom_and_non_oom_are_rejected(self) -> None:
            for status in ("FAIL_OOM", "FAIL_NON_OOM"):
                with self.subTest(status=status):
                    raw = self.passing()
                    raw["status"] = status
                    self.assertTrue(raw_pass_blockers(raw, self.expected()))

        def test_other_machine_is_rejected(self) -> None:
            raw = self.passing()
            raw["gpu_uuid"] = "GPU-other"
            self.assertTrue(raw_pass_blockers(raw, self.expected()))

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PreflightContractTests)
    )
    return 0 if result.wasSuccessful() else 1


if MODE == "--self-test":
    raise SystemExit(run_self_tests())
if MODE != "run":
    print(f"ERROR: unsupported argument: {MODE}", file=sys.stderr)
    raise SystemExit(2)

if platform.system() != "Linux":
    print("ERROR: this preflight is Linux-only.", file=sys.stderr)
    raise SystemExit(2)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "custom_models/src"))

from benchmark_v2.hardware_preflight import (  # noqa: E402
    machine_identity,
    preflight_artifact_path,
    preflight_identity,
)
from benchmark_v2.training_profiles import load_training_profile  # noqa: E402
from scripts.uniform_batch4_machine_gate import current_freeze_identity  # noqa: E402

profile = load_training_profile(PROFILE_ID)
if profile.profile_hash != PROFILE_HASH:
    print("ERROR: frozen training profile hash mismatch.", file=sys.stderr)
    raise SystemExit(2)

machine = machine_identity()
if not machine.get("gpu_name"):
    print("ERROR: CUDA GPU identity is unavailable.", file=sys.stderr)
    raise SystemExit(2)
freeze = current_freeze_identity()
if not freeze.get("git_commit"):
    print(
        "ERROR: Git commit unavailable; set UNIFORM_BATCH4_GIT_COMMIT for a source archive.",
        file=sys.stderr,
    )
    raise SystemExit(2)

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

if summary_path.is_file() and raw_path.is_file():
    raw = json_load(raw_path)
    raw_blockers = raw_pass_blockers(raw, expected)
    existing_blockers = summary_blockers(
        json_load(summary_path),
        expected=expected,
        machine=machine,
        freeze=freeze,
        raw_path=raw_path,
        raw_sha256=sha256_file(raw_path),
    )
    if not raw_blockers and not existing_blockers:
        print(f"PASS already exists and matches this Linux machine: {summary_path}")
        raise SystemExit(0)
    print("ERROR: existing current-machine PASS summary is stale or mismatched:", file=sys.stderr)
    for blocker in [*raw_blockers, *existing_blockers]:
        print(f"- {blocker}", file=sys.stderr)
    print("Refusing to overwrite it.", file=sys.stderr)
    raise SystemExit(3)
if summary_path.exists():
    print(f"ERROR: refusing to overwrite existing summary path: {summary_path}", file=sys.stderr)
    raise SystemExit(3)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
log_dir = LOG_ROOT / str(machine["machine_id"]) / f"{stamp}_{os.getpid()}"
log_dir.mkdir(parents=True, exist_ok=False)
log_path = log_dir / "preflight.log"
command = [
    PYTHON_BIN,
    str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
    "hardware-preflight",
    "--model",
    MODEL_ID,
    "--training-profile",
    PROFILE_ID,
    "--preflight-root",
    str(RAW_ROOT),
]
print("Launching isolated Original-loss MSGNet preflight:")
print(" ".join(command))
with log_path.open("x", encoding="utf-8") as log:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
(log_dir / "preflight_exit_code.txt").write_text(
    f"{completed.returncode}\n", encoding="utf-8"
)

if not raw_path.is_file():
    print(f"ERROR: preflight produced no artifact: {raw_path}", file=sys.stderr)
    raise SystemExit(completed.returncode or 3)
raw = json_load(raw_path)
blockers = raw_pass_blockers(raw, expected)
if completed.returncode != 0 or blockers:
    print(
        f"MSGNet Original preflight did not PASS on this machine; raw artifact preserved: {raw_path}",
        file=sys.stderr,
    )
    for blocker in blockers:
        print(f"- {blocker}", file=sys.stderr)
    raise SystemExit(completed.returncode or 3)

raw_sha256 = sha256_file(raw_path)
summary = {
    "schema_version": "uniform_batch4_original_msgnet_preflight_v1",
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
    "raw_preflight_sha256": raw_sha256,
    "preflight_log": str(log_path),
    "checks": {
        "forward": True,
        "original_formal_loss": True,
        "backward": True,
        "finite_loss": True,
        "finite_gradients": True,
        "prediction_shape_4_134_10": True,
    },
    "created_at": utc_now(),
}
write_new_json(summary_path, summary)
print(f"PASS: {summary_path}")
raise SystemExit(0)
PYTHON
exit $?
