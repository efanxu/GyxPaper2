from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "custom_models/src"
if str(SOURCE_ROOT) not in sys.path: sys.path.insert(0, str(SOURCE_ROOT))
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v2.artifacts import atomic_write_json
from benchmark_v2.experiments.e5_common_loss.a8_reference import validate_a8_reference
from benchmark_v2.experiments.e5_common_loss.contracts import NONTRAINABLE_MODELS, TRAINABLE_MODELS
from benchmark_v2.experiments.e5_common_loss.readiness import (
    build_readiness as build_common_readiness,
    inspect_benchmark_run,
)
from benchmark_v2.process_lock import (
    ProcessLockError,
    acquire_lock as acquire_process_lock,
    clear_stale_lock as clear_process_stale_lock,
    lock_status as process_lock_status,
    release_lock as release_process_lock,
)
from st_mgprompt.a8_batch4_contract import A8_REFERENCE_ID


DEFAULT_MANIFEST = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
CURRENT_RUN_MAP = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_RUN_ID_MAP.json"
EXPECTED_OUTPUT_ROOT = PROJECT_ROOT / "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026"
AUDIT_ROOT = PROJECT_ROOT / "custom_models/logs/uniform_bs4/audit/e5_scope27"
LOCK_PATH = PROJECT_ROOT / "custom_models/logs/uniform_bs4/e5_scope27.lock.json"
ARCHIVE_DIRECTORY = "_archived_attempts"
QUARANTINE_DIRECTORY = "_quarantine"
E5_SCOPE_ID = "e5_batch4_scope27_seed2026"
HORIZONS = (3, 6, 10)


class ScopeGateError(RuntimeError): pass


def utc_now() -> str: return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = load_json(Path(path))
    if not isinstance(payload, dict): raise ScopeGateError("E5 manifest must be an object.")
    return payload


def _load_run_map(path: Path = CURRENT_RUN_MAP) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict): raise ScopeGateError("E5 run map must be an object.")
    return payload


def _entry_sets(manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    entries = list(manifest.get("entries", []))
    return {
        "trainable": tuple(row["model_id"] for row in entries if row.get("entry_type") == "TRAIN_COMMON_LOSS"),
        "evaluate_only": tuple(row["model_id"] for row in entries if row.get("entry_type") == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"),
        "a8_reference": tuple(row["model_id"] for row in entries if row.get("model_id") == "st_mgprompt_a8"),
    }


def validate_manifest(manifest: Mapping[str, Any], run_map: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    entries = list(manifest.get("entries", [])); groups = _entry_sets(manifest); reasons = []
    if manifest.get("scope_id") != E5_SCOPE_ID: reasons.append("SCOPE_ID_CONFLICT")
    if (len(groups["trainable"]), len(groups["evaluate_only"]), len(groups["a8_reference"]), len(entries)) != (24, 2, 1, 27): reasons.append("COUNT_CONFLICT")
    if set(groups["trainable"]) != set(TRAINABLE_MODELS): reasons.append("TRAINABLE_SET_CONFLICT")
    if set(groups["evaluate_only"]) != set(NONTRAINABLE_MODELS): reasons.append("EVALUATE_ONLY_SET_CONFLICT")
    ids = [row.get("model_id") for row in entries]; run_ids = [row.get("e5_run_id") or row.get("run_id") for row in entries]
    if len(set(ids)) != 27 or len(set(run_ids)) != 27: reasons.append("DUPLICATE_MODEL_OR_RUN_ID")
    if {"segrnn", "msgnet"} & set(ids): reasons.append("EXCLUDED_MODEL_PRESENT")
    if any(row.get("loss_id") != "masked_score_aligned_hybrid" for row in entries): reasons.append("LOSS_CONFLICT")
    if manifest.get("training_profile_id") != "uniform_train_batch4_v1": reasons.append("TRAINING_PROFILE_CONFLICT")
    benchmark = [row for row in entries if row.get("entry_type") != "REFERENCE_ONLY_FORMAL_A8"]
    if any(not str(row.get("e5_run_id", "")).endswith("_bs4_seed2026") for row in benchmark): reasons.append("NON_BATCH4_RUN_ID")
    references = [row for row in entries if row.get("entry_type") == "REFERENCE_ONLY_FORMAL_A8"]
    if len(references) != 1 or references[0].get("e5_run_id") != A8_REFERENCE_ID: reasons.append("A8_REFERENCE_ID_CONFLICT")
    selected = dict(run_map or _load_run_map()); mapped = {row.get("model_id"): row.get("e5_run_id") or row.get("run_id") for row in selected.get("entries", [])}
    if mapped and any(mapped.get(row["model_id"]) != (row.get("e5_run_id") or row.get("run_id")) for row in entries): reasons.append("RUN_MAP_CONFLICT")
    if reasons: raise ScopeGateError(f"E5 manifest invalid: {reasons}")
    return {"status": "PASS", "scope_id": E5_SCOPE_ID, "counts": {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27}, "excluded": ["segrnn", "msgnet"], "loss_id": "masked_score_aligned_hybrid"}


def compute_e5_freeze(manifest: Mapping[str, Any], *, run_map: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    validate_manifest(manifest, run_map=run_map)
    entries = [{key: row.get(key) for key in ("ordinal", "entry_id", "entry_type", "model_id", "e5_run_id", "output_root", "loss_id", "precision_identity", "formal_training") if key in row} for row in manifest["entries"]]
    return {"schema_version": "e5_scope27_explicit_snapshot_v2", "scope_id": E5_SCOPE_ID, "protocol_version": manifest.get("schema_version"), "batch": 4, "seed": 2026, "lookback": 144, "max_pred_len": 10, "eval_horizons": [3, 6, 10], "loss_id": "masked_score_aligned_hybrid", "counts": {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27}, "entries": entries}


def _entry(manifest: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for row in manifest["entries"]:
        if row.get("model_id") == model_id: return row
    raise ScopeGateError(f"Unknown E5 model: {model_id}")


def inspect_run(manifest: Mapping[str, Any], entry: Mapping[str, Any], output_root: Path, **_: Any) -> dict[str, Any]:
    if entry["model_id"] == "st_mgprompt_a8": return inspect_a8()
    run_id = entry.get("e5_run_id") or entry.get("run_id"); run_dir = output_root / str(run_id)
    return inspect_benchmark_run(entry, run_dir)


def inspect_trainable_run(manifest: Mapping[str, Any], entry: Mapping[str, Any], output_root: Path, **kwargs: Any) -> dict[str, Any]: return inspect_run(manifest, entry, output_root, **kwargs)
def inspect_evaluate_only_run(manifest: Mapping[str, Any], entry: Mapping[str, Any], output_root: Path, **kwargs: Any) -> dict[str, Any]: return inspect_run(manifest, entry, output_root, **kwargs)


def inspect_a8() -> dict[str, Any]:
    reference = validate_a8_reference()
    ready = reference["status"] == "READY"
    return {"model_id": "st_mgprompt_a8", "run_id": A8_REFERENCE_ID, "entry_type": "REFERENCE_ONLY_FORMAL_A8", "status": "COMPLETED" if ready else "NOT_READY", "ready": ready, "reasons": [] if ready else [*reference["metrics_validation_reasons"], "A8_REFERENCE_NOT_READY"], "metrics_complete": reference["metrics_complete"], "checkpoint_loadable": reference["checkpoint_loadable"], "explicit_config_conflicts": reference["config_conflicts"], "reference": reference}


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return pid > 0 and psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception: return False


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception: return None


def lock_status(path: Path = LOCK_PATH) -> dict[str, Any]:
    return process_lock_status(path, scope_id=E5_SCOPE_ID)


def acquire_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return acquire_process_lock(path, scope_id=E5_SCOPE_ID)


def release_lock(owner: Mapping[str, Any], path: Path = LOCK_PATH) -> bool:
    return release_process_lock(path, owner=owner)


def clear_stale_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return clear_process_stale_lock(path, scope_id=E5_SCOPE_ID)


def _active_worker(run_dir: Path, inspected: Mapping[str, Any]) -> bool:
    del run_dir
    pid = inspected.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid): return False
    expected = inspected.get("process_start_time"); actual = _process_start_time(pid)
    return expected is None or actual is None or abs(float(expected) - actual) < 1


def _plan_action(entry: Mapping[str, Any], inspected: Mapping[str, Any]) -> str:
    if entry.get("entry_type") == "REFERENCE_ONLY_FORMAL_A8":
        return "SKIP_COMPLETED" if inspected.get("ready") else "BLOCK_A8_REFERENCE_NOT_READY"
    if inspected.get("status") == "RUN_MISSING": return "RUN_MISSING"
    if inspected.get("explicit_config_conflicts"): return "BLOCK_EXPLICIT_CONFIG_CONFLICT"
    if inspected.get("ready"): return "SKIP_COMPLETED"
    if _active_worker(Path(inspected.get("run_dir", EXPECTED_OUTPUT_ROOT)), inspected): return "BLOCK_ACTIVE_PROCESS"
    return "ARCHIVE_AND_RUN"


def build_plan(manifest: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    validate_manifest(manifest); rows = []
    for entry in manifest["entries"]:
        inspected = inspect_run(manifest, entry, output_root)
        rows.append({**inspected, "action": _plan_action(entry, inspected)})
    return {"status": "PASS", "scope_id": E5_SCOPE_ID, "counts": {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27}, "entries": rows}


def build_preflight_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    return {"status": "PASS", "scope_id": E5_SCOPE_ID, "required": 24, "entries": [{"model_id": row["model_id"], "run_id": row.get("e5_run_id"), "batch_size": 4, "loss_id": row["loss_id"]} for row in manifest["entries"] if row["entry_type"] == "TRAIN_COMMON_LOSS"]}


def run_preflight_suite(manifest: Mapping[str, Any], *, preflight_root: Path,
                        report_path: Path, child_log_root: Path) -> tuple[int, dict[str, Any]]:
    validate_manifest(manifest)
    a8 = inspect_a8()
    if not a8.get("ready"):
        return 74, {"status": "A8_PREREQUISITE_NOT_READY", "scope_id": E5_SCOPE_ID, "a8": a8}
    child_log_root.mkdir(parents=True, exist_ok=True)
    from benchmark_v2.hardware_preflight import launch_preflight, read_matching_pass
    results = []
    for entry in manifest["entries"]:
        if entry["entry_type"] != "TRAIN_COMMON_LOSS": continue
        log_path = child_log_root / f"{int(entry['ordinal']):02d}_{entry['model_id']}.log"
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            def child_runner(argv: Sequence[str], check: bool = False, *, _handle: Any = handle) -> Any:
                return subprocess.run(argv, cwd=str(PROJECT_ROOT),
                                      env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
                                      stdout=_handle, stderr=subprocess.STDOUT, check=check)
            code = launch_preflight(entry["model_id"], root=preflight_root,
                                    experiment_profile="e5_common_loss_v1",
                                    training_profile="uniform_train_batch4_v1",
                                    formal_scope_id=E5_SCOPE_ID,
                                    run_id=entry["e5_run_id"], runner=child_runner)
        matched = read_matching_pass(
            entry["model_id"], root=preflight_root, experiment_profile="e5_common_loss_v1",
            training_profile="uniform_train_batch4_v1", formal_scope_id=E5_SCOPE_ID,
            run_id=entry["e5_run_id"])
        passed = code == 0 and matched is not None
        results.append({"status": "PASS" if passed else "FAILED", "scope_id": E5_SCOPE_ID,
                        "model_id": entry["model_id"], "run_id": entry["e5_run_id"],
                        "batch_size": 4, "precision": None if matched is None else matched.get("precision"), "device": "cuda",
                        "forward_pass": False if matched is None else matched.get("forward_pass"),
                        "backward_pass": False if matched is None else matched.get("backward_pass"),
                        "finite": False if matched is None else matched.get("finite"),
                        "output_shape": None if matched is None else matched.get("output_shape"), "created_at": utc_now(),
                        "exit_code": int(code), "log_path": str(log_path)})
    passed_count = sum(row["status"] == "PASS" for row in results)
    payload = {"schema_version": "e5_scope27_preflight_run_v2",
               "status": "PASS" if passed_count == 24 else "COMPLETED_WITH_FAILURES",
               "scope_id": E5_SCOPE_ID, "counts": {"pass": passed_count, "expected": 24},
               "entries": results}
    atomic_write_json(report_path, payload)
    return (0 if passed_count == 24 else 1), payload


def _run_command(entry: Mapping[str, Any], *, input_path: str | None,
                 target_path: str | None, preflight_root: Path) -> list[str]:
    action = "evaluate-only" if entry["entry_type"] == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC" else "train"
    run_id = str(entry.get("e5_run_id") or entry.get("run_id"))
    args = [sys.executable, str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
            action, "--model", str(entry["model_id"]), "--run-id", run_id,
            "--output-root", str(EXPECTED_OUTPUT_ROOT), "--experiment-profile", "e5_common_loss_v1",
            "--training-profile", "uniform_train_batch4_v1", "--formal-scope-id", E5_SCOPE_ID]
    if input_path: args.extend(["--input-path", input_path])
    if target_path: args.extend(["--target-path", target_path])
    if action == "train": args.extend(["--preflight-root", str(preflight_root)])
    return args


def run_scope(manifest: Mapping[str, Any], *, input_path: str | None, target_path: str | None,
              log_root: Path, preflight_root: Path,
              suite_report_path: Path | None = None,
              runner: Any = subprocess.run) -> tuple[int, dict[str, Any]]:
    state = lock_status()
    if state["status"] != "ABSENT":
        return 74, {"status": f"LOCK_{state['status']}", "scope_id": E5_SCOPE_ID, "lock": state}
    plan = build_plan(manifest, EXPECTED_OUTPUT_ROOT)
    blocked = [row for row in plan["entries"] if row["action"] in {"BLOCK_EXPLICIT_CONFIG_CONFLICT", "BLOCK_ACTIVE_PROCESS", "BLOCK_A8_REFERENCE_NOT_READY"}]
    if blocked: return 74, {"status": "BLOCKED", "scope_id": E5_SCOPE_ID, "entries": blocked}
    a8 = inspect_a8()
    if not a8.get("ready"): return 74, {"status": "A8_PREREQUISITE_NOT_READY", "a8": a8}
    from benchmark_v2.hardware_preflight import read_matching_pass
    missing = [entry["model_id"] for entry, row in zip(manifest["entries"], plan["entries"])
               if entry["entry_type"] == "TRAIN_COMMON_LOSS" and row["action"] != "SKIP_COMPLETED"
               and read_matching_pass(entry["model_id"], root=preflight_root,
                                      experiment_profile="e5_common_loss_v1",
                                      training_profile="uniform_train_batch4_v1",
                                      formal_scope_id=E5_SCOPE_ID,
                                      run_id=entry["e5_run_id"]) is None]
    if missing: return 74, {"status": "PREFLIGHT_MISSING", "scope_id": E5_SCOPE_ID, "models": missing}
    try:
        owner = acquire_lock()
    except ProcessLockError:
        raced = lock_status()
        return 74, {"status": f"LOCK_{raced['status']}", "scope_id": E5_SCOPE_ID, "lock": raced}
    results = []
    try:
        log_root.mkdir(parents=True, exist_ok=True)
        for entry, row in zip(manifest["entries"], plan["entries"]):
            if entry["model_id"] == "st_mgprompt_a8":
                results.append({"model_id": entry["model_id"], "run_id": A8_REFERENCE_ID, "status": "REFERENCE_READY", "exit_code": 0})
                continue
            if row["action"] == "SKIP_COMPLETED":
                results.append({"model_id": entry["model_id"], "run_id": entry.get("e5_run_id"), "status": "SKIP_COMPLETED", "exit_code": 0})
                continue
            log_path = log_root / f"{int(entry['ordinal']):02d}_{entry['model_id']}.log"
            try:
                if row["action"] == "ARCHIVE_AND_RUN":
                    archive_or_quarantine(manifest, entry, EXPECTED_OUTPUT_ROOT, apply=True)
                command = _run_command(entry, input_path=input_path, target_path=target_path, preflight_root=preflight_root)
                with log_path.open("w", encoding="utf-8", newline="") as handle:
                    completed = runner(command, cwd=str(PROJECT_ROOT),
                                       env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
                                       stdout=handle, stderr=subprocess.STDOUT, check=False)
                exit_code = int(completed.returncode)
                results.append({"model_id": entry["model_id"], "run_id": entry.get("e5_run_id"),
                                "status": "COMPLETED" if exit_code == 0 else "FAILED",
                                "exit_code": exit_code, "log_path": str(log_path)})
            except Exception as exc:
                results.append({"model_id": entry["model_id"], "run_id": entry.get("e5_run_id"),
                                "status": "FAILED", "exit_code": 1,
                                "error_type": type(exc).__name__, "error_message": str(exc),
                                "log_path": str(log_path)})
        readiness = build_readiness(manifest, EXPECTED_OUTPUT_ROOT)
        code = 0 if len(results) == 27 and all(row["exit_code"] == 0 for row in results) and readiness["status"] == "READY" else 1
        payload = {"schema_version": "e5_scope27_suite_report_v1",
                   "status": "COMPLETED" if code == 0 else "COMPLETED_WITH_FAILURES",
                   "scope_id": E5_SCOPE_ID, "entry_count": len(results),
                   "entries": results, "readiness": readiness}
        atomic_write_json(suite_report_path or AUDIT_ROOT / "e5_scope27_suite_report.json", payload)
        return code, payload
    finally:
        release_lock(owner)


def build_readiness(manifest: Mapping[str, Any], output_root: Path, **kwargs: Any) -> dict[str, Any]:
    validate_manifest(manifest)
    options: dict[str, Any] = {}
    if kwargs.get("a8_run_root") is not None:
        options["a8_run_root"] = kwargs["a8_run_root"]
    return build_common_readiness(
        manifest=manifest,
        output_root=output_root,
        training_profile="uniform_train_batch4_v1",
        a8_reference_path=kwargs.get("a8_reference_path") or (
            AUDIT_ROOT / "E5_A8_BATCH4_REFERENCE.json"
        ),
        **options,
    )


def write_readiness_outputs(report: Mapping[str, Any], report_path: Path, evidence_path: Path) -> None:
    atomic_write_json(report_path, dict(report))
    atomic_write_json(evidence_path, dict(report))


def aggregate(manifest: Mapping[str, Any], output_root: Path, *, require_complete: bool,
              report_path: Path, evidence_path: Path) -> dict[str, Any]:
    readiness = build_readiness(manifest, output_root)
    write_readiness_outputs(readiness, report_path, evidence_path)
    if readiness["status"] != "READY":
        raise ScopeGateError(f"E5_RESULT_NOT_READY:{readiness['ready_entries']}/27")
    from benchmark_v2.experiments.e5_common_loss.aggregation import aggregate as aggregate_results

    return aggregate_results(
        output_root=output_root,
        require_complete=require_complete,
        training_profile="uniform_train_batch4_v1",
        manifest=manifest,
        a8_reference_path=AUDIT_ROOT / "E5_A8_BATCH4_REFERENCE.json",
    )


def archive_or_quarantine(manifest: Mapping[str, Any], entry: Mapping[str, Any], output_root: Path, *, apply: bool = False, kind: str = ARCHIVE_DIRECTORY, **_: Any) -> dict[str, Any]:
    if entry.get("entry_type") == "REFERENCE_ONLY_FORMAL_A8":
        raise ScopeGateError("A8 reference is read-only and cannot be archived.")
    run_id = entry.get("e5_run_id") or entry.get("run_id"); source = output_root / str(run_id); target = output_root / kind / f"{run_id}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    inspected = inspect_benchmark_run(entry, source)
    if _active_worker(source, inspected):
        raise ScopeGateError("BLOCK_ACTIVE_PROCESS: active run directory cannot be archived.")
    files = [path for path in source.rglob("*") if path.is_file()] if source.is_dir() else []
    result = {"schema_version": "e5_archive_receipt_v2", "scope_id": E5_SCOPE_ID, "model_id": entry["model_id"], "run_id": run_id, "action": kind, "source": str(source), "target": str(target), "status": "PREVIEW", "started_at": utc_now(), "finished_at": None, "exit_code": None, "file_count": len(files), "total_size_bytes": sum(path.stat().st_size for path in files), "message": "Existing attempt will be preserved."}
    if apply and source.is_dir(): target.parent.mkdir(parents=True, exist_ok=True); os.replace(source, target); result.update({"status": "COMPLETED", "finished_at": utc_now(), "exit_code": 0})
    return result


def inventory(manifest: Mapping[str, Any], output_root: Path) -> dict[str, Any]: return build_plan(manifest, output_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-manifest", "freeze-plan", "freeze", "preflight-plan", "plan-runs", "dry-run", "static-audit", "readiness", "inventory", "lock-status"):
        item = sub.add_parser(name)
        if name in {"plan-runs", "dry-run", "static-audit", "readiness", "inventory"}: item.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT))
    clear = sub.add_parser("clear-stale-lock"); clear.add_argument("--lock-path", default=str(LOCK_PATH))
    preflight = sub.add_parser("preflight"); preflight.add_argument("--preflight-root", default=str(AUDIT_ROOT / "preflight")); preflight.add_argument("--source-revision"); preflight.add_argument("--report-path", default=str(AUDIT_ROOT / "e5_scope27_preflight_summary.json")); preflight.add_argument("--child-log-root", default=str(AUDIT_ROOT / "preflight/child_logs"))
    run = sub.add_parser("run"); run.add_argument("--input-path"); run.add_argument("--target-path"); run.add_argument("--log-root", default=str(AUDIT_ROOT / "runs")); run.add_argument("--preflight-root", default=str(AUDIT_ROOT / "preflight")); run.add_argument("--suite-report-path", default=str(AUDIT_ROOT / "e5_scope27_suite_report.json")); run.add_argument("--source-revision")
    aggregate_parser = sub.add_parser("aggregate"); aggregate_parser.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT)); aggregate_parser.add_argument("--report-path", default=str(AUDIT_ROOT / "e5_scope27_readiness.json")); aggregate_parser.add_argument("--evidence-path", default=str(AUDIT_ROOT / "e5_scope27_evidence.json")); aggregate_parser.add_argument("--require-complete", action="store_true")
    for name in ("readiness", "inventory"):
        sub.choices[name].add_argument("--report-path")
    sub.choices["readiness"].add_argument("--evidence-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv); manifest = load_manifest(args.manifest)
    try:
        if args.command == "validate-manifest": result = validate_manifest(manifest)
        elif args.command in {"freeze-plan", "freeze"}: result = compute_e5_freeze(manifest)
        elif args.command == "preflight-plan": result = build_preflight_plan(manifest)
        elif args.command in {"plan-runs", "dry-run", "static-audit"}: result = build_plan(manifest, Path(args.output_root))
        elif args.command == "readiness": result = build_readiness(manifest, Path(args.output_root))
        elif args.command == "inventory": result = inventory(manifest, Path(args.output_root))
        elif args.command == "lock-status": result = lock_status()
        elif args.command == "clear-stale-lock": result = clear_stale_lock(Path(args.lock_path))
        elif args.command == "preflight":
            code, result = run_preflight_suite(manifest, preflight_root=Path(args.preflight_root), report_path=Path(args.report_path), child_log_root=Path(args.child_log_root)); print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return code
        elif args.command == "run":
            code, result = run_scope(manifest, input_path=args.input_path, target_path=args.target_path, log_root=Path(args.log_root), preflight_root=Path(args.preflight_root), suite_report_path=Path(args.suite_report_path)); print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return code
        elif args.command == "aggregate":
            result = aggregate(manifest, Path(args.output_root), require_complete=args.require_complete, report_path=Path(args.report_path), evidence_path=Path(args.evidence_path))
        else: raise ScopeGateError(f"Unsupported command: {args.command}")
        if args.command in {"readiness", "inventory"} and args.report_path: atomic_write_json(Path(args.report_path), result)
        if args.command == "readiness" and args.evidence_path: atomic_write_json(Path(args.evidence_path), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 4 if args.command == "readiness" and result.get("status") != "READY" else 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error_message": str(exc)}, ensure_ascii=False)); return 2


if __name__ == "__main__": raise SystemExit(main())
