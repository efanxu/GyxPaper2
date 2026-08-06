from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ArtifactError
from .registry import load_registry


PROFILES = {
    "TRAIN": {"resolved_config.json", "effective_config.json", "protocol_check.json", "environment.json", "data_signature.json", "model_summary.json", "artifact_manifest.json", "best_checkpoint.pt", "last_checkpoint.pt", "train_log.csv", "metrics_eval_h3.json", "metrics_eval_h6.json", "metrics_eval_h10.json", "metrics.csv", "prediction_metadata.json", "run_status.json"},
    "EVALUATE_ONLY": {"resolved_config.json", "effective_config.json", "protocol_check.json", "environment.json", "data_signature.json", "model_summary.json", "artifact_manifest.json", "metrics_eval_h3.json", "metrics_eval_h6.json", "metrics_eval_h10.json", "metrics.csv", "prediction_metadata.json", "run_status.json"},
    "NON_TRAINABLE": {"resolved_config.json", "effective_config.json", "protocol_check.json", "environment.json", "data_signature.json", "model_summary.json", "artifact_manifest.json", "baseline_state.json", "metrics_eval_h3.json", "metrics_eval_h6.json", "metrics_eval_h10.json", "metrics.csv", "prediction_metadata.json", "run_status.json"},
    "REFERENCE_ONLY": {"resolved_config.json", "effective_config.json", "protocol_check.json", "artifact_manifest.json", "run_status.json"},
    "SMOKE": {"resolved_config.json", "effective_config.json", "protocol_check.json", "model_summary.json", "artifact_manifest.json", "best_checkpoint.pt", "last_checkpoint.pt", "train_log.csv", "metrics_eval_h3.json", "metrics_eval_h6.json", "metrics_eval_h10.json", "metrics.csv", "prediction_metadata.json", "run_status.json"},
    "FAILED": {"run_status.json"},
}

STATUSES = {"CREATED", "PROTOCOL_VALIDATED", "RUNNING", "EARLY_STOPPED", "TRAINING_COMPLETED", "EVALUATING", "COMPLETED", "FAILED", "REFERENCE_ONLY"}
GRAPH_METADATA_KEYS = (
    "graph_id",
    "node_count",
    "ordered_node_ids",
    "selected_k",
    "graph_support_names",
    "graph_support_shapes",
    "graph_runtime_dtype",
    "graph_context_id",
    "uses_physical_support",
    "physical_support_names",
    "physical_support_shapes",
    "adaptive_graph_policy",
    "node_identity_policy",
    "temporal_identity_policy",
    "adaptive_initialization_policy",
)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def safe_run_dir(root: str | Path, *parts: str, allow_existing: bool = False) -> Path:
    root = Path(root).resolve()
    run_dir = (root.joinpath(*parts)).resolve()
    if root != run_dir and root not in run_dir.parents:
        raise ArtifactError("Run path escapes its configured artifact root")
    if run_dir.exists() and not allow_existing:
        raise ArtifactError(f"Run directory already exists; refusing overwrite: {run_dir}")
    return run_dir


def write_status(run_dir: str | Path, *, status: str, run_mode: str, artifact_profile: str, **extra: Any) -> None:
    if status not in STATUSES:
        raise ArtifactError(f"Unknown run status {status}")
    path = Path(run_dir) / "run_status.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    if status == "RUNNING":
        import psutil

        extra.setdefault("pid", os.getpid())
        extra.setdefault("process_start_time", float(psutil.Process(os.getpid()).create_time()))
    effective_path = Path(run_dir) / "effective_config.json"
    if effective_path.is_file():
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
        for key in GRAPH_METADATA_KEYS:
            if key in effective:
                extra.setdefault(key, effective[key])
    terminal = status in {"EARLY_STOPPED", "TRAINING_COMPLETED", "COMPLETED", "FAILED", "REFERENCE_ONLY"}
    payload = {"status": status, "run_mode": run_mode, "artifact_profile": artifact_profile, "started_at": extra.pop("started_at", previous.get("started_at") or now), "finished_at": extra.pop("finished_at", now if terminal else previous.get("finished_at")), "exit_code": extra.pop("exit_code", 0 if status in {"COMPLETED", "REFERENCE_ONLY", "EARLY_STOPPED", "TRAINING_COMPLETED"} else None), "failure_stage": extra.pop("failure_stage", None), "error_type": extra.pop("error_type", None), "error_message": extra.pop("error_message", None), "traceback_tail": extra.pop("traceback_tail", None), **extra}
    atomic_write_json(Path(run_dir) / "run_status.json", payload)


def validate_run(
    run_dir: str | Path,
    *,
    expected_training_batch_profile_id: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        raise ArtifactError("run_status.json is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    profile = status.get("artifact_profile")
    if profile not in PROFILES:
        raise ArtifactError(f"Unknown artifact profile: {profile}")
    missing = sorted(name for name in PROFILES[profile] if not (run_dir / name).exists())
    if missing:
        raise ArtifactError(f"Incomplete {profile} artifact; missing: {missing}")
    if expected_training_batch_profile_id is not None:
        manifest = json.loads(
            (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        effective = json.loads(
            (run_dir / "effective_config.json").read_text(encoding="utf-8")
        )
        for payload in (manifest, effective):
            if (
                expected_training_batch_profile_id is not None
                and payload.get("training_batch_profile_id")
                != expected_training_batch_profile_id
            ):
                raise ArtifactError("BLOCKED_MIXED_BATCH_PROFILE: profile id mismatch")
    effective_path = run_dir / "effective_config.json"
    if effective_path.exists():
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
        expected_graph = {
            key: effective[key]
            for key in GRAPH_METADATA_KEYS
            if key in effective
        }
        if expected_graph:
            for name in (
                "resolved_config.json",
                "protocol_check.json",
                "model_summary.json",
                "artifact_manifest.json",
                "prediction_metadata.json",
                "run_status.json",
            ):
                candidate = run_dir / name
                if not candidate.exists():
                    continue
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                mismatches = {
                    key: (payload.get(key), expected)
                    for key, expected in expected_graph.items()
                    if payload.get(key) != expected
                }
                if mismatches:
                    raise ArtifactError(
                        f"{name} graph metadata mismatch: {mismatches}"
                    )
    return {"status": "PASS", "run_dir": str(run_dir), "artifact_profile": profile, "run_status": status.get("status"), "missing": []}


def is_formal_discoverable(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir).resolve()
    formal_root = (Path(__file__).resolve().parents[3] / "results" / "benchmark_v2").resolve()
    try:
        run_dir.relative_to(formal_root)
    except ValueError:
        return False
    try:
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        registry = load_registry()
        model_id = json.loads((run_dir / "effective_config.json").read_text(encoding="utf-8")).get("model_id")
        return status.get("run_mode") == "formal" and status.get("status") == "COMPLETED" and model_id in {e.canonical_id for e in registry.list()} and model_id != "__framework_test_only__" and all((run_dir / f"metrics_eval_h{h}.json").exists() for h in (3, 6, 10))
    except (OSError, ValueError, KeyError, ArtifactError):
        return False
