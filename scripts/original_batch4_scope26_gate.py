from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "custom_models/src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_v2.artifacts import atomic_write_json
from benchmark_v2.original_scope26 import CURRENT_SCOPE26_ID, load_current_scope_manifest


CURRENT_MANIFEST = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json"
CURRENT_RUN_MAP = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_SCOPE26_RUN_ID_MAP.json"
RESULT_ROOT = PROJECT_ROOT / "custom_models/results/benchmark_v2_uniform_bs4"
AUDIT_ROOT = PROJECT_ROOT / "custom_models/logs/uniform_bs4/audit"
LOCK_PATH = PROJECT_ROOT / "custom_models/logs/uniform_bs4/original_scope26.lock.json"
ARCHIVE_DIRECTORY = "_archived_attempts"
QUARANTINE_DIRECTORY = "_quarantine"
HORIZONS = (3, 6, 10)


class Scope26GateError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"MISSING:{path.name}"
    except (OSError, ValueError) as exc:
        return None, f"INVALID:{path.name}:{type(exc).__name__}"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _load_run_map(path: Path = CURRENT_RUN_MAP) -> dict[str, Any]:
    payload, error = _read_json(path)
    if error or not isinstance(payload, dict):
        raise Scope26GateError(error or "Run map must be an object.")
    return payload


def validate_manifest(manifest: Mapping[str, Any], run_map: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    entries = list(manifest.get("entries", []))
    trainable = [row for row in entries if row.get("entry_type") == "TRAINABLE"]
    evaluate_only = [row for row in entries if row.get("entry_type") == "EVALUATE_ONLY"]
    ids = [str(row.get("model_id")) for row in entries]
    run_ids = [str(row.get("run_id")) for row in entries]
    reasons = []
    if manifest.get("scope_id") != CURRENT_SCOPE26_ID:
        reasons.append("SCOPE_ID_CONFLICT")
    if (len(trainable), len(evaluate_only), len(entries)) != (24, 2, 26):
        reasons.append("COUNT_CONFLICT")
    if len(set(ids)) != 26 or len(set(run_ids)) != 26:
        reasons.append("DUPLICATE_MODEL_OR_RUN_ID")
    if {"segrnn", "msgnet"} & set(ids):
        reasons.append("EXCLUDED_MODEL_PRESENT")
    if int(manifest.get("seed", -1)) != 2026 or int(manifest.get("lookback", -1)) != 144 or int(manifest.get("prediction_horizon", -1)) != 10:
        reasons.append("PROTOCOL_CONFIG_CONFLICT")
    selected_map = dict(run_map or _load_run_map())
    mapped = {row.get("model_id"): row.get("run_id") for row in selected_map.get("entries", [])}
    if mapped and any(mapped.get(row["model_id"]) != row["run_id"] for row in entries):
        reasons.append("RUN_MAP_CONFLICT")
    if reasons:
        raise Scope26GateError(f"Original manifest invalid: {reasons}")
    return {"status": "PASS", "scope_id": CURRENT_SCOPE26_ID, "counts": {"trainable": 24, "evaluate_only": 2, "total": 26}, "excluded": ["segrnn", "msgnet"]}


def compute_original_freeze(manifest: Mapping[str, Any], *, run_map: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    validate_manifest(manifest, run_map=run_map)
    entries = [{key: row.get(key) for key in ("ordinal", "entry_id", "entry_type", "model_id", "run_id", "output_root", "artifact_profile", "formal_training", "precision_identity") if key in row} for row in manifest["entries"]]
    return {
        "schema_version": "original_scope26_explicit_snapshot_v2",
        "scope_id": CURRENT_SCOPE26_ID, "protocol_version": manifest.get("schema_version"),
        "seed": 2026, "batch": 4, "gradient_accumulation_steps": 1,
        "lookback": 144, "max_pred_len": 10, "eval_horizons": [3, 6, 10],
        "counts": {"trainable": 24, "evaluate_only": 2, "total": 26},
        "entries": entries,
    }


def _metric_payloads(run_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    payloads, reasons = {}, []
    for horizon in HORIZONS:
        path = run_dir / f"metrics_eval_h{horizon}.json"
        payload, error = _read_json(path)
        if error or not isinstance(payload, dict):
            reasons.append(error or f"INVALID:{path.name}")
            continue
        payloads[horizon] = payload
        if payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_CONFLICT:{horizon}")
        for field in ("MAE", "RMSE", "R2", "Score"):
            if not _finite(payload.get(field)):
                reasons.append(f"NONFINITE:H{horizon}:{field}")
        count = payload.get("valid_target_count", payload.get("ValidCount"))
        if not _finite(count) or float(count) <= 0:
            reasons.append(f"INVALID_COUNT:H{horizon}")
    return payloads, reasons


def _expected_entry(manifest: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for row in manifest["entries"]:
        if row.get("model_id") == model_id:
            return row
    raise Scope26GateError(f"Unknown model: {model_id}")


def _loadable_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        import torch
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def _entry_root(entry: Mapping[str, Any], output_root: Path) -> Path:
    selected = Path(output_root).resolve()
    if selected == RESULT_ROOT.resolve() and entry.get("output_root"):
        return (PROJECT_ROOT / str(entry["output_root"])).resolve()
    return selected


def inspect_run(manifest: Mapping[str, Any], entry: Mapping[str, Any], output_root: Path, **_: Any) -> dict[str, Any]:
    run_dir = _entry_root(entry, output_root) / str(entry["run_id"])
    if not run_dir.is_dir():
        return {"model_id": entry["model_id"], "run_id": entry["run_id"], "run_dir": str(run_dir), "status": "RUN_MISSING", "reasons": ["RUN_MISSING"], "explicit_config_conflicts": []}
    status, status_error = _read_json(run_dir / "run_status.json")
    effective, effective_error = _read_json(run_dir / "effective_config.json")
    reasons = [value for value in (status_error, effective_error) if value]
    status = status if isinstance(status, dict) else {}
    effective = effective if isinstance(effective, dict) else {}
    expected = {
        "model_id": entry["model_id"], "run_id": entry["run_id"],
        "scope_id": manifest["scope_id"], "seed": 2026,
        "artifact_profile": entry.get("artifact_profile"),
        "formal_training": bool(entry.get("formal_training")),
    }
    conflicts = []
    aliases = {"scope_id": ("scope_id", "active_scope_id"), "model_id": ("model_id",), "run_id": ("run_id",), "seed": ("seed",), "artifact_profile": ("artifact_profile",), "formal_training": ("formal_training",)}
    for key, expected_value in expected.items():
        values = [effective.get(alias) for alias in aliases[key] if alias in effective]
        if values and any(value != expected_value for value in values):
            conflicts.append(key)
    _, metric_reasons = _metric_payloads(run_dir)
    completed = status.get("status") == "COMPLETED" and status.get("exit_code") == 0
    checkpoint_ok = True if entry["entry_type"] == "EVALUATE_ONLY" else _loadable_checkpoint(run_dir / "best_checkpoint.pt")
    if not completed:
        reasons.append("NOT_COMPLETED")
    reasons.extend(metric_reasons)
    if not checkpoint_ok:
        reasons.append("CHECKPOINT_NOT_LOADABLE")
    if conflicts:
        reasons.append("EXPLICIT_CONFIG_CONFLICT")
    return {
        "model_id": entry["model_id"], "run_id": entry["run_id"], "run_dir": str(run_dir),
        "status": "COMPLETED" if completed and not metric_reasons and checkpoint_ok and not conflicts else str(status.get("status") or "INCOMPLETE"),
        "reasons": reasons, "explicit_config_conflicts": conflicts,
        "metrics_complete": not metric_reasons, "checkpoint_loadable": checkpoint_ok,
        "pid": status.get("pid"), "process_start_time": status.get("process_start_time"),
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _active_worker(run_dir: Path, inspected: Mapping[str, Any]) -> bool:
    del run_dir
    pid = inspected.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False
    expected = inspected.get("process_start_time")
    actual = _process_start_time(pid)
    return expected is None or actual is None or abs(float(expected) - actual) < 1.0


def _plan_action(entry: Mapping[str, Any], inspected: Mapping[str, Any], **_: Any) -> str:
    if inspected["status"] == "RUN_MISSING":
        return "RUN_MISSING"
    if inspected.get("explicit_config_conflicts"):
        return "BLOCK_EXPLICIT_CONFIG_CONFLICT"
    if inspected["status"] == "COMPLETED":
        return "SKIP_COMPLETED"
    if _active_worker(Path(inspected["run_dir"]), inspected):
        return "BLOCK_ACTIVE_PROCESS"
    return "ARCHIVE_AND_RUN"


def archive_existing_attempt(entry: Mapping[str, Any], output_root: Path, *, apply: bool = False, kind: str = ARCHIVE_DIRECTORY, **_: Any) -> dict[str, Any]:
    base = _entry_root(entry, output_root)
    source = base / str(entry["run_id"])
    target = base / kind / f"{entry['run_id']}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    files = [path for path in source.rglob("*") if path.is_file()] if source.is_dir() else []
    result = {"schema_version": "archive_receipt_v2", "scope_id": CURRENT_SCOPE26_ID, "model_id": entry["model_id"], "run_id": entry["run_id"], "action": "archive", "source": str(source), "target": str(target), "status": "PREVIEW", "started_at": _utc_now(), "finished_at": None, "exit_code": None, "file_count": len(files), "total_size_bytes": sum(path.stat().st_size for path in files), "message": "Existing attempt will be preserved."}
    if apply:
        if not source.is_dir():
            result.update({"status": "SOURCE_MISSING", "finished_at": _utc_now(), "exit_code": 0})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            result.update({"status": "COMPLETED", "finished_at": _utc_now(), "exit_code": 0})
    return result


def quarantine_existing(manifest: Mapping[str, Any], model_id: str, output_root: Path, *, apply: bool = False, **kwargs: Any) -> dict[str, Any]:
    return archive_existing_attempt(_expected_entry(manifest, model_id), output_root, apply=apply, kind=QUARANTINE_DIRECTORY, **kwargs)


def build_plan(manifest: Mapping[str, Any], output_root: Path, **_: Any) -> dict[str, Any]:
    validate_manifest(manifest)
    rows = []
    for entry in manifest["entries"]:
        inspected = inspect_run(manifest, entry, output_root)
        rows.append({**inspected, "action": _plan_action(entry, inspected)})
    return {"scope_id": CURRENT_SCOPE26_ID, "status": "PASS", "counts": {"total": 26, "trainable": 24, "evaluate_only": 2}, "entries": rows}


def build_preflight_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    entries = [{"model_id": row["model_id"], "run_id": row["run_id"], "batch_size": 4, "scope_id": CURRENT_SCOPE26_ID} for row in manifest["entries"] if row["entry_type"] == "TRAINABLE"]
    return {"status": "PASS", "scope_id": CURRENT_SCOPE26_ID, "required": 24, "entries": entries}


def run_preflight_suite(manifest: Mapping[str, Any], *, preflight_root: Path,
                        report_path: Path, child_log_root: Path) -> tuple[int, dict[str, Any]]:
    validate_manifest(manifest)
    child_log_root.mkdir(parents=True, exist_ok=True)
    from benchmark_v2.hardware_preflight import launch_preflight, read_matching_pass

    results = []
    for entry in manifest["entries"]:
        if entry["entry_type"] != "TRAINABLE":
            continue
        log_path = child_log_root / f"{int(entry['ordinal']):02d}_{entry['model_id']}.log"
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            def child_runner(argv: Sequence[str], check: bool = False, *, _handle: Any = handle) -> Any:
                return subprocess.run(argv, cwd=str(PROJECT_ROOT),
                                      env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
                                      stdout=_handle, stderr=subprocess.STDOUT, check=check)
            code = launch_preflight(
                entry["model_id"], root=preflight_root,
                training_profile="uniform_train_batch4_v1",
                formal_scope_id=CURRENT_SCOPE26_ID,
                run_id=entry["run_id"], runner=child_runner,
            )
        passed = code == 0 and read_matching_pass(
            entry["model_id"], root=preflight_root,
            training_profile="uniform_train_batch4_v1",
            formal_scope_id=CURRENT_SCOPE26_ID,
            run_id=entry["run_id"],
        ) is not None
        results.append({
            "status": "PASS" if passed else "FAILED", "scope_id": CURRENT_SCOPE26_ID,
            "model_id": entry["model_id"], "run_id": entry["run_id"],
            "batch_size": 4, "precision": entry.get("precision_identity"),
            "device": entry.get("device", "cuda"), "forward_pass": passed,
            "backward_pass": passed, "finite": passed, "output_shape": [4, 134, 10],
            "created_at": _utc_now(), "exit_code": int(code), "log_path": str(log_path),
        })
    passed_count = sum(row["status"] == "PASS" for row in results)
    payload = {"schema_version": "original_scope26_preflight_run_v2",
               "status": "PASS" if passed_count == 24 else "COMPLETED_WITH_FAILURES",
               "scope_id": CURRENT_SCOPE26_ID, "counts": {"pass": passed_count, "expected": 24},
               "entries": results}
    atomic_write_json(report_path, payload)
    return (0 if passed_count == 24 else 1), payload


def _run_command(entry: Mapping[str, Any], *, input_path: str | None,
                 target_path: str | None, preflight_root: Path) -> list[str]:
    command = "evaluate-only" if entry["entry_type"] == "EVALUATE_ONLY" else "train"
    args = [sys.executable, str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
            command, "--model", str(entry["model_id"]), "--run-id", str(entry["run_id"]),
            "--output-root", str(PROJECT_ROOT / str(entry["output_root"])),
            "--training-profile", "uniform_train_batch4_v1", "--formal-scope-id", CURRENT_SCOPE26_ID]
    if input_path: args.extend(["--input-path", input_path])
    if target_path: args.extend(["--target-path", target_path])
    if command == "train": args.extend(["--preflight-root", str(preflight_root)])
    return args


def run_suite(manifest: Mapping[str, Any], *, input_path: str | None, target_path: str | None,
              log_root: Path, preflight_root: Path) -> tuple[int, dict[str, Any]]:
    plan = build_plan(manifest, RESULT_ROOT)
    blocked = [row for row in plan["entries"] if row["action"] in {"BLOCK_EXPLICIT_CONFIG_CONFLICT", "BLOCK_ACTIVE_PROCESS"}]
    if blocked: return 74, {"status": "BLOCKED", "scope_id": CURRENT_SCOPE26_ID, "entries": blocked}
    from benchmark_v2.hardware_preflight import read_matching_pass
    missing = [entry["model_id"] for entry, row in zip(manifest["entries"], plan["entries"])
               if entry["entry_type"] == "TRAINABLE" and row["action"] != "SKIP_COMPLETED"
               and read_matching_pass(entry["model_id"], root=preflight_root,
                                      training_profile="uniform_train_batch4_v1",
                                      formal_scope_id=CURRENT_SCOPE26_ID,
                                      run_id=entry["run_id"]) is None]
    if missing: return 74, {"status": "PREFLIGHT_MISSING", "scope_id": CURRENT_SCOPE26_ID, "models": missing}
    log_root.mkdir(parents=True, exist_ok=True); results = []
    for entry, row in zip(manifest["entries"], plan["entries"]):
        if row["action"] == "SKIP_COMPLETED":
            results.append({"model_id": entry["model_id"], "run_id": entry["run_id"], "status": "SKIP_COMPLETED", "exit_code": 0}); continue
        if row["action"] == "ARCHIVE_AND_RUN": archive_existing_attempt(entry, RESULT_ROOT, apply=True)
        command = _run_command(entry, input_path=input_path, target_path=target_path, preflight_root=preflight_root)
        log_path = log_root / f"{int(entry['ordinal']):02d}_{entry['model_id']}.log"
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT),
                                       env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
                                       stdout=handle, stderr=subprocess.STDOUT, check=False)
        results.append({"model_id": entry["model_id"], "run_id": entry["run_id"],
                        "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
                        "exit_code": int(completed.returncode), "log_path": str(log_path)})
    readiness = build_readiness(manifest, RESULT_ROOT)
    code = 0 if all(row["exit_code"] == 0 for row in results) and readiness["status"] == "READY" else 1
    return code, {"status": "COMPLETED" if code == 0 else "COMPLETED_WITH_FAILURES",
                  "scope_id": CURRENT_SCOPE26_ID, "entries": results, "readiness": readiness}


def aggregate(manifest: Mapping[str, Any], *, output_root: Path, require_complete: bool) -> tuple[int, dict[str, Any]]:
    if not require_complete: return 2, {"status": "AGGREGATE_SKIPPED_NOT_COMPLETE", "reason": "--require-complete is mandatory"}
    readiness = build_readiness(manifest, output_root)
    if readiness["status"] != "READY": return 4, {"status": "AGGREGATE_SKIPPED_NOT_COMPLETE", "readiness": readiness}
    return 0, {"status": "PASS", "scope_id": CURRENT_SCOPE26_ID,
               "counts": {"ready": 26, "expected": 26}, "result_root": str(output_root)}


def _lock_status(path: Path = LOCK_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "ABSENT", "path": str(path)}
    payload, error = _read_json(path)
    required = {"scope_id", "hostname", "pid", "process_start_time", "created_at"}
    if error or not isinstance(payload, dict) or not required.issubset(payload):
        return {"status": "MALFORMED", "path": str(path)}
    if payload["hostname"] != socket.gethostname():
        return {"status": "STALE", "path": str(path), "owner": payload}
    pid = payload.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return {"status": "STALE", "path": str(path), "owner": payload}
    actual = _process_start_time(pid)
    if actual is not None and abs(float(payload["process_start_time"]) - actual) >= 1.0:
        return {"status": "STALE", "path": str(path), "owner": payload}
    return {"status": "ACTIVE", "path": str(path), "owner": payload}


def clear_stale_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    status = _lock_status(path)
    if status["status"] == "STALE":
        path.unlink()
        return {"status": "CLEARED", "path": str(path)}
    return status


def build_readiness(manifest: Mapping[str, Any], output_root: Path, **_: Any) -> dict[str, Any]:
    plan = build_plan(manifest, output_root)
    ready = sum(row["action"] == "SKIP_COMPLETED" for row in plan["entries"])
    return {"scope_id": CURRENT_SCOPE26_ID, "status": "READY" if ready == 26 else "NOT_READY", "ready_entries": ready, "expected_entries": 26, "entries": plan["entries"]}


def write_inventory(manifest: Mapping[str, Any], root: Path, json_path: Path, csv_path: Path) -> dict[str, Any]:
    report = build_plan(manifest, root)
    atomic_write_json(json_path, report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", "run_id", "status", "action"])
        writer.writeheader()
        writer.writerows(report["entries"])
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(CURRENT_MANIFEST))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-manifest", "freeze", "preflight-plan", "dry-run", "static-audit", "readiness", "lock-status"):
        item = sub.add_parser(name)
        if name in {"dry-run", "static-audit", "readiness"}:
            item.add_argument("--output-root", default=str(RESULT_ROOT))
    clear = sub.add_parser("clear-stale-lock"); clear.add_argument("--lock-path", default=str(LOCK_PATH))
    preflight = sub.add_parser("preflight"); preflight.add_argument("--preflight-root", default=str(AUDIT_ROOT / "preflight/original_scope26")); preflight.add_argument("--source-revision"); preflight.add_argument("--report-path", default=str(AUDIT_ROOT / "original_scope26_preflight_summary.json")); preflight.add_argument("--child-log-root", default=str(AUDIT_ROOT / "original_scope26_preflight_child_logs"))
    run = sub.add_parser("run"); run.add_argument("--input-path"); run.add_argument("--target-path"); run.add_argument("--log-root", default=str(AUDIT_ROOT / "runs")); run.add_argument("--preflight-root", default=str(AUDIT_ROOT / "preflight/original_scope26")); run.add_argument("--source-revision")
    agg = sub.add_parser("aggregate"); agg.add_argument("--output-root", default=str(RESULT_ROOT)); agg.add_argument("--require-complete", action="store_true")
    quarantine = sub.add_parser("quarantine-existing"); quarantine.add_argument("--model", required=True); quarantine.add_argument("--output-root", default=str(RESULT_ROOT)); quarantine.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_current_scope_manifest(args.manifest)
    try:
        if args.command == "validate-manifest": result = validate_manifest(manifest)
        elif args.command == "freeze": result = compute_original_freeze(manifest)
        elif args.command == "preflight-plan": result = build_preflight_plan(manifest)
        elif args.command in {"dry-run", "static-audit"}: result = build_plan(manifest, Path(args.output_root))
        elif args.command == "readiness": result = build_readiness(manifest, Path(args.output_root))
        elif args.command == "lock-status": result = _lock_status()
        elif args.command == "clear-stale-lock": result = clear_stale_lock(Path(args.lock_path))
        elif args.command == "preflight":
            code, result = run_preflight_suite(manifest, preflight_root=Path(args.preflight_root), report_path=Path(args.report_path), child_log_root=Path(args.child_log_root)); print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return code
        elif args.command == "run":
            code, result = run_suite(manifest, input_path=args.input_path, target_path=args.target_path, log_root=Path(args.log_root), preflight_root=Path(args.preflight_root)); print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return code
        elif args.command == "aggregate":
            code, result = aggregate(manifest, output_root=Path(args.output_root), require_complete=args.require_complete); print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return code
        elif args.command == "quarantine-existing": result = quarantine_existing(manifest, args.model, Path(args.output_root), apply=args.apply)
        else: raise Scope26GateError(f"Unsupported command: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 4 if args.command == "readiness" and result.get("status") != "READY" else 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error_message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
