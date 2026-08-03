from __future__ import annotations

import hashlib
import json
import math
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...artifacts import atomic_write_json
from ...runtime import PROJECT_ROOT
from ...training_profiles import load_training_profile
from .contracts import A8_REFERENCE_ID, BATCH4_A8_REFERENCE_ID
from .loss_profile import (
    BENCHMARK_PROTOCOL_HASH,
    CLI_PROFILE_ID,
    get_profile_metadata,
)


SUITE_ROOT = (
    PROJECT_ROOT
    / "custom_models/results/st_mgprompt_component_ablation"
    / "component_ablation_fixed_dual_seed2026"
)
A8_ROOT = SUITE_ROOT / "A8"
A8_RUN_ROOT = A8_ROOT / "STMGPrompt_ComponentAblation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_batch4_metrics(run_root: Path) -> tuple[bool, list[str], str | None]:
    paths = [run_root / f"metrics_eval_h{h}.json" for h in (3, 6, 10)]
    payloads: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    for path, horizon in zip(paths, (3, 6, 10)):
        if not path.is_file():
            reasons.append(f"MISSING:{path.name}")
            continue
        try:
            payload = _load(path)
        except Exception as exc:
            reasons.append(f"INVALID:{path.name}:{type(exc).__name__}")
            continue
        if not isinstance(payload, dict):
            reasons.append(f"NOT_OBJECT:{path.name}")
            continue
        payloads[horizon] = payload
        if payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_MISMATCH:{path.name}")
        for key in ("MAE", "RMSE", "R2", "Score"):
            value = payload.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                reasons.append(f"NONFINITE:{path.name}:{key}")
        count = payload.get("valid_target_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            reasons.append(f"INVALID_VALID_TARGET_COUNT:{path.name}")
    csv_path = run_root / "metrics.csv"
    if not csv_path.is_file():
        reasons.append("MISSING:metrics.csv")
    else:
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except Exception as exc:
            rows = []
            reasons.append(f"INVALID:metrics.csv:{type(exc).__name__}")
        by_horizon = {}
        for row in rows:
            try:
                by_horizon[int(row.get("horizon", ""))] = row
            except (TypeError, ValueError):
                continue
        for horizon, payload in payloads.items():
            row = by_horizon.get(horizon)
            if row is None:
                reasons.append(f"CSV_MISSING_H{horizon}")
                continue
            for key in ("MAE", "RMSE", "R2", "Score"):
                try:
                    value = float(row.get(key))
                except (TypeError, ValueError):
                    value = float("nan")
                if not math.isfinite(value) or not math.isclose(
                    value, float(payload[key]), rel_tol=1e-9, abs_tol=1e-9
                ):
                    reasons.append(f"JSON_CSV_MISMATCH_H{horizon}_{key}")
            try:
                count = int(row.get("valid_target_count"))
            except (TypeError, ValueError):
                count = None
            if count != payload.get("valid_target_count"):
                reasons.append(f"JSON_CSV_MISMATCH_H{horizon}_valid_target_count")
            for key in ("MAE", "RMSE", "R2", "Score"):
                if row.get(key) in {None, ""}:
                    reasons.append(f"CSV_NONFINITE_H{horizon}_{key}")
    if not all(path.is_file() for path in paths):
        return False, reasons, None
    digest = _combined_hash([*paths, csv_path]) if csv_path.is_file() else None
    return not reasons, reasons, digest


def validate_a8_reference(
    training_profile: str | None = None,
) -> dict[str, Any]:
    batch_profile = load_training_profile(training_profile)
    if batch_profile is not None:
        batch4_run_root = (
            PROJECT_ROOT
            / "custom_models/results/st_mgprompt_uniform_bs4"
            / "component_ablation_a8_bs4_seed2026"
            / "STMGPrompt_ComponentAblation"
        )
        required = [
            batch4_run_root / "effective_config.json",
            batch4_run_root / "effective_config_diff.json",
            batch4_run_root / "training_batch_profile.json",
            batch4_run_root / "artifact_manifest.json",
            batch4_run_root / "best_checkpoint.pt",
            batch4_run_root / "run_status.json",
            batch4_run_root / "train_complete.json",
            batch4_run_root / "evaluation_complete.json",
            batch4_run_root / "prediction_metadata.json",
            batch4_run_root / "protocol_check.json",
            batch4_run_root / "metrics.csv",
            *[
                batch4_run_root / f"metrics_eval_h{h}.json"
                for h in (3, 6, 10)
            ],
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return {
                "status": "BLOCKED_FORMAL_A8_NOT_READY",
                "missing": missing,
                "reference_id": BATCH4_A8_REFERENCE_ID,
                **batch_profile.identity(),
            }
        config = _load(batch4_run_root / "effective_config.json")
        run_status = _load(batch4_run_root / "run_status.json")
        train = _load(batch4_run_root / "train_complete.json")
        evaluation = _load(batch4_run_root / "evaluation_complete.json")
        protocol_check = _load(batch4_run_root / "protocol_check.json")
        expected_config = {
            "component_ablation": "A8",
            "loss_function": "masked_score_aligned_hybrid",
            "loss_protocol": "fair_main",
            "use_msmg_dwu": False,
            "seed": 2026,
            "lookback": 144,
            "max_pred_len": 10,
            "definition": "w/o MS-MG-DWU",
            "training_profile": "uniform_train_batch4_v1",
            "train_batch_size": 4,
            "val_batch_size": 4,
            "test_batch_size": 4,
            **batch_profile.identity(),
        }
        mismatches = {
            key: {"actual": config.get(key), "expected": value}
            for key, value in expected_config.items()
            if config.get(key) != value
        }
        metrics_paths = [
            batch4_run_root / f"metrics_eval_h{h}.json"
            for h in (3, 6, 10)
        ]
        metrics_complete, metric_reasons, metrics_hash = _validate_batch4_metrics(
            batch4_run_root
        )
        checkpoint = batch4_run_root / "best_checkpoint.pt"
        checkpoint_nonempty = checkpoint.is_file() and checkpoint.stat().st_size > 0
        artifact_manifest = _load(batch4_run_root / "artifact_manifest.json")
        config_diff = _load(batch4_run_root / "effective_config_diff.json")
        formal_complete = (
            run_status.get("status") == "COMPLETED"
            and run_status.get("exit_code") == 0
            and run_status.get("current_stage") == "PROCESS_FINISHED"
            and train.get("status") == "completed"
            and evaluation.get("status") == "completed"
            and protocol_check.get("passed") is True
            and metrics_complete
            and checkpoint_nonempty
            and config_diff.get("passed") is True
        )
        profile = get_profile_metadata(CLI_PROFILE_ID)
        passed = (
            formal_complete
            and not mismatches
            and not metric_reasons
            and artifact_manifest.get("retrained_in_e5") is False
            and artifact_manifest.get("checkpoint_copied") is False
            and artifact_manifest.get("metrics_copied") is False
        )
        return {
            "reference_type": "FORMAL_A8_BATCH4_READ_ONLY",
            "reference_id": BATCH4_A8_REFERENCE_ID,
            "status": "VALID" if passed else "BLOCKED_FORMAL_A8_NOT_READY",
            "source_absolute_or_resolved_path": str(batch4_run_root.resolve()),
            "source_relative_path": batch4_run_root.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "source_run_id": "component_ablation_a8_bs4_seed2026",
            "source_variant": "A8",
            "source_definition": "w/o MS-MG-DWU",
            "source_checkpoint_path": str(
                (batch4_run_root / "best_checkpoint.pt").resolve()
            ),
            "source_checkpoint_sha256": _sha256(checkpoint),
            "source_metrics_paths": [
                str(path.resolve()) for path in metrics_paths
            ],
            "source_metrics_sha256": metrics_hash,
            "source_config_path": str(
                (batch4_run_root / "effective_config.json").resolve()
            ),
            "source_config_sha256": _sha256(
                batch4_run_root / "effective_config.json"
            ),
            "source_protocol_hash": BENCHMARK_PROTOCOL_HASH,
            "source_protocol_evidence_path": str(
                (batch4_run_root / "protocol_check.json").resolve()
            ),
            "source_protocol_evidence_sha256": _sha256(
                batch4_run_root / "protocol_check.json"
            ),
            "source_loss_id": config.get("loss_function"),
            "source_loss_hash": profile["loss_source_hash"],
            "loss_profile_hash": profile["loss_profile_hash"],
            "trained_with_common_loss": True,
            "formal_complete": formal_complete,
            "metrics_complete": metrics_complete,
            "metrics_validation_reasons": metric_reasons,
            "protocol_match": not mismatches,
            "config_mismatches": mismatches,
            "checkpoint_copied": False,
            "metrics_copied": False,
            "retrained_in_e5": artifact_manifest.get("retrained_in_e5"),
            "checkpoint_sha256": _sha256(checkpoint),
            **batch_profile.identity(),
            "reference_created_at": datetime.now(timezone.utc).isoformat(),
        }
    required = [
        SUITE_ROOT / "experiment_manifest.json",
        SUITE_ROOT / "experiment_status.json",
        A8_ROOT / "effective_config.json",
        A8_ROOT / "effective_config_diff.json",
        A8_RUN_ROOT / "best_checkpoint.pt",
        A8_RUN_ROOT / "run_status.json",
        A8_RUN_ROOT / "train_complete.json",
        A8_RUN_ROOT / "evaluation_complete.json",
        A8_RUN_ROOT / "prediction_metadata.json",
        A8_RUN_ROOT / "protocol_check.json",
        *[A8_RUN_ROOT / f"metrics_eval_h{h}.json" for h in (3, 6, 10)],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "status": "BLOCKED_FORMAL_A8_NOT_READY",
            "missing": missing,
            "reference_id": A8_REFERENCE_ID,
        }
    suite_manifest = _load(SUITE_ROOT / "experiment_manifest.json")
    suite_status = _load(SUITE_ROOT / "experiment_status.json")
    config = _load(A8_ROOT / "effective_config.json")
    config_diff = _load(A8_ROOT / "effective_config_diff.json")
    run_status = _load(A8_RUN_ROOT / "run_status.json")
    train = _load(A8_RUN_ROOT / "train_complete.json")
    evaluation = _load(A8_RUN_ROOT / "evaluation_complete.json")
    protocol_check = _load(A8_RUN_ROOT / "protocol_check.json")
    a8_definition = next(
        (item for item in suite_manifest["variants"] if item["variant_id"] == "A8"),
        None,
    )
    a8_suite_status = next(
        (item for item in suite_status if item["variant"] == "A8"), None
    )
    expected_config = {
        "component_ablation": "A8",
        "loss_function": "masked_score_aligned_hybrid",
        "loss_protocol": "fair_main",
        "use_msmg_dwu": False,
        "seed": 2026,
        "smoke": False,
        "lookback": 144,
        "max_pred_len": 10,
        "eval_horizons": [3, 6, 10],
        "split_ratios": [0.8, 0.1, 0.1],
        "train_sample_stride": 6,
        "val_sample_stride": 3,
        "test_sample_stride": 1,
        "train_batch_size": 32,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "target_col": "Patv_raw",
        "input_patv_col": "Patv_clean_for_input",
        "target_mask_col": "valid_target_mask",
        "checkpoint_selection_metric": "val_official_score_h10",
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    metrics_paths = [A8_RUN_ROOT / f"metrics_eval_h{h}.json" for h in (3, 6, 10)]
    metrics_complete = all(
        _load(path).get("horizon") == horizon
        and all(key in _load(path) for key in ("Score", "MAE", "RMSE", "R2"))
        for path, horizon in zip(metrics_paths, (3, 6, 10))
    )
    formal_complete = (
        a8_definition is not None
        and a8_definition.get("display_name") == "w/o MS-MG-DWU"
        and a8_suite_status is not None
        and a8_suite_status.get("status") == "COMPLETED"
        and a8_suite_status.get("returncode") == 0
        and run_status.get("current_stage") == "PROCESS_FINISHED"
        and train.get("status") == "completed"
        and evaluation.get("status") == "completed"
        and protocol_check.get("passed") is True
        and config_diff.get("passed") is True
        and metrics_complete
    )
    profile = get_profile_metadata(CLI_PROFILE_ID)
    passed = formal_complete and not mismatches
    reference = {
        "reference_type": "FORMAL_A8_READ_ONLY",
        "reference_id": A8_REFERENCE_ID,
        "status": "VALID" if passed else "BLOCKED_FORMAL_A8_NOT_READY",
        "source_absolute_or_resolved_path": str(A8_RUN_ROOT.resolve()),
        "source_relative_path": A8_RUN_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "source_run_id": "component_ablation_fixed_dual_seed2026/A8",
        "source_variant": "A8",
        "source_definition": "w/o MS-MG-DWU",
        "source_checkpoint_path": str((A8_RUN_ROOT / "best_checkpoint.pt").resolve()),
        "source_checkpoint_sha256": _sha256(A8_RUN_ROOT / "best_checkpoint.pt"),
        "source_metrics_paths": [str(path.resolve()) for path in metrics_paths],
        "source_metrics_sha256": _combined_hash(metrics_paths),
        "source_config_path": str((A8_ROOT / "effective_config.json").resolve()),
        "source_config_sha256": _sha256(A8_ROOT / "effective_config.json"),
        "source_protocol_evidence_path": str(
            (A8_RUN_ROOT / "protocol_check.json").resolve()
        ),
        "source_protocol_evidence_sha256": _sha256(
            A8_RUN_ROOT / "protocol_check.json"
        ),
        "source_protocol_hash": BENCHMARK_PROTOCOL_HASH if not mismatches else None,
        "source_loss_id": config.get("loss_function"),
        "source_loss_hash": profile["loss_source_hash"],
        "loss_profile_hash": profile["loss_profile_hash"],
        "trained_with_common_loss": True,
        "retrained_in_e5": False,
        "comparison_role": "ST_MGPROMPT_STRUCTURE_REFERENCE",
        "formal_complete": formal_complete,
        "metrics_complete": metrics_complete,
        "protocol_match": not mismatches,
        "config_mismatches": mismatches,
        "checkpoint_copied": False,
        "metrics_copied": False,
        "reference_created_at": datetime.now(timezone.utc).isoformat(),
    }
    return reference


def create_a8_reference(
    output_path: str | Path, training_profile: str | None = None
) -> dict[str, Any]:
    reference = validate_a8_reference(training_profile=training_profile)
    atomic_write_json(Path(output_path), reference)
    return reference
