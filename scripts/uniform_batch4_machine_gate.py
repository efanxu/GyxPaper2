from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "custom_models/src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts import original_batch4_scope26_gate as original_gate
from scripts import st_mgprompt_a8_batch4_gate as a8_gate


PROFILE_ID = "uniform_train_batch4_v1"


def current_freeze_identity() -> dict[str, Any]:
    """Return the explicit Batch4 snapshot retained for older callers."""
    original = original_gate.load_current_scope_manifest(original_gate.CURRENT_MANIFEST)
    return {
        "schema_version": "uniform_batch4_explicit_snapshot_v1",
        "training_batch_profile_id": PROFILE_ID,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "seed": 2026,
        "lookback": 144,
        "max_pred_len": 10,
        "eval_horizons": [3, 6, 10],
        "original": original_gate.validate_manifest(original),
        "a8": a8_gate.validate_contract(),
    }


def precheck() -> dict[str, Any]:
    snapshot = current_freeze_identity()
    return {"status": "PASS", **snapshot}


def preflight_inventory(suite: str) -> dict[str, Any]:
    if suite == "original":
        manifest = original_gate.load_current_scope_manifest(original_gate.CURRENT_MANIFEST)
        return original_gate.build_preflight_plan(manifest)
    raise ValueError(f"Unknown benchmark suite: {suite}")


def st_preflight_inventory(suite: str) -> dict[str, Any]:
    a8 = a8_gate.build_preflight_plan()
    if suite == "a8":
        return a8
    return {
        "status": "PLANNED",
        "scope": "full",
        "a8": a8,
        "original": preflight_inventory("original"),
    }


def verify(suite: str) -> dict[str, Any]:
    if suite == "original":
        manifest = original_gate.load_current_scope_manifest(original_gate.CURRENT_MANIFEST)
        return original_gate.build_readiness(manifest, original_gate.RESULT_ROOT)
    if suite == "a8":
        return a8_gate.build_readiness()
    return {
        "status": "REPORT",
        "original": verify("original"),
        "a8": verify("a8"),
    }


def formal_guidance(suite: str) -> dict[str, Any]:
    launchers = {
        "original": "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_SCOPE26_WINDOWS_FORMAL_COMMANDS.ps1",
        "a8": "custom_models/docs/benchmark_v2/A8/A8_BATCH4_LINUX.sh",
    }
    return {
        "status": "USE_SCOPE_LAUNCHER",
        "suite": suite,
        "launcher": launchers[suite],
        "message": "The legacy wrapper is informational; use the maintained scope launcher.",
    }


def compare_manifests(left: Path, right: Path) -> dict[str, Any]:
    left_payload = json.loads(left.read_text(encoding="utf-8"))
    right_payload = json.loads(right.read_text(encoding="utf-8"))
    return {
        "status": "MATCH" if left_payload == right_payload else "DIFFERENT",
        "left": str(left),
        "right": str(right),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("precheck")
    inventory = sub.add_parser("preflight-inventory")
    inventory.add_argument("--suite", choices=("original",), required=True)
    st = sub.add_parser("preflight-st")
    st.add_argument("--suite", choices=("full", "a8"), required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--suite", choices=("original", "full", "a8"), required=True)
    formal = sub.add_parser("formal-exec")
    formal.add_argument("--suite", choices=("original", "a8"), required=True)
    formal.add_argument("--python")
    compare = sub.add_parser("compare-manifests")
    compare.add_argument("left")
    compare.add_argument("right")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "precheck": payload = precheck()
    elif args.command == "preflight-inventory": payload = preflight_inventory(args.suite)
    elif args.command == "preflight-st": payload = st_preflight_inventory(args.suite)
    elif args.command == "verify": payload = verify(args.suite)
    elif args.command == "formal-exec": payload = formal_guidance(args.suite)
    else: payload = compare_manifests(Path(args.left), Path(args.right))
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
