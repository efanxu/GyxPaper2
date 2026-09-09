from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .config import STMGPromptConfig
from .artifact_status import inspect_variant_artifacts, is_completed_status
from .experiment_protocol import (
    CANONICAL_ID,
    COMPONENT_ABLATION_VARIANTS,
    COMPONENT_RESULT_ROOT,
    COMPONENT_RUN_ID,
    PRECISION_RESULT_ROOT,
    PRECISION_RUN_ID,
    PRECISION_VARIANTS,
    apply_variant,
    assert_expected_diff,
    canonical_directory,
    config_diff,
    get_variant,
    semantic_config,
    write_json,
    write_reference,
    write_variant_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT = Path(__file__).with_name("run_st_mgprompt.py")


def get_current_python_executable() -> Path:
    """Return the executable that started this runner."""

    python_executable = Path(sys.executable).resolve()
    if not python_executable.is_file():
        raise FileNotFoundError(
            "Current Python interpreter does not exist: "
            f"{python_executable}"
        )
    return python_executable


def normalize_python_command(command: Sequence[str]) -> list[str]:
    """Replace any stale/generated interpreter at command[0] with sys.executable."""

    if not command:
        raise RuntimeError("Generated subprocess command is empty.")
    python_executable = get_current_python_executable()
    return [str(python_executable), *[str(item) for item in command[1:]]]


def _validate_python_override(value: str) -> None:
    """Reject --python values that could silently select another environment."""

    requested = Path(value).expanduser().resolve()
    current = get_current_python_executable()
    if not requested.is_file():
        raise FileNotFoundError(f"Requested --python interpreter does not exist: {requested}")
    if requested != current:
        raise ValueError(
            "--python must point to the interpreter running this process. "
            f"Requested: {requested}; current: {current}"
        )
    print(f"Child Python selection validated: {requested}", flush=True)


def parser_for(family: str) -> argparse.ArgumentParser:
    label = "P0-P5 precision" if family == "precision" else "A0-A9 component"
    parser = argparse.ArgumentParser(description=f"Run formal Fixed-Dual {label} experiments.")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python executable used for child processes. Defaults to the "
            "interpreter running the current process; an explicit value must "
            "exist and resolve to that same interpreter."
        ),
    )
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--full-shape-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip variants whose train and evaluation artifacts already pass audit.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=None)
    return parser


def _family_data(family: str):
    if family == "precision":
        return PRECISION_VARIANTS, PRECISION_RESULT_ROOT, PRECISION_RUN_ID, "P0"
    if family == "component_ablation":
        return COMPONENT_ABLATION_VARIANTS, COMPONENT_RESULT_ROOT, COMPONENT_RUN_ID, "A0"
    raise ValueError(f"Unknown experiment family: {family}")


def selected_variants(family: str, names: list[str] | None):
    mapping, _, _, _ = _family_data(family)
    requested = [str(name).upper() for name in (names or list(mapping))]
    variants = []
    for name in requested:
        variants.append(get_variant(name, family))
    return variants


def _root(args: argparse.Namespace, family: str) -> Path:
    _, default_root, default_run_id, _ = _family_data(family)
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.run_smoke:
        output_root = PROJECT_ROOT / "custom_models/results_smoke/st_mgprompt_fixed_dual_refactor"
    elif args.full_shape_smoke:
        output_root = (
            PROJECT_ROOT
            / "custom_models/results_smoke/st_mgprompt_fixed_dual_refactor_full_shape"
        )
    else:
        output_root = PROJECT_ROOT / default_root
    run_id = args.run_id or default_run_id
    return output_root / run_id


def _effective_artifacts(root: Path, variant) -> tuple[dict[str, Any], dict[str, Any]]:
    config = apply_variant(STMGPromptConfig(), variant.variant_id, variant.experiment_family)
    config.validate()
    effective = config.to_dict()
    diff = assert_expected_diff(config, variant.variant_id, variant.experiment_family)
    variant_dir = root / variant.variant_id
    write_json(variant_dir / "effective_config.json", effective)
    write_json(variant_dir / "effective_config_diff.json", diff)
    return effective, diff


def _command(
    args: argparse.Namespace,
    family: str,
    root: Path,
    variant,
    *,
    evaluate_only: bool = False,
    resume: bool = False,
) -> list[str]:
    if not variant.trainable:
        raise ValueError(f"{variant.variant_id} is REFERENCE_ONLY.")
    mode = "--full"
    output_root = root.parent
    run_id = f"{root.name}/{variant.variant_id}"
    if args.run_smoke:
        mode = "--smoke"
    elif args.full_shape_smoke:
        mode = "--full-shape-smoke"
    command = [
        str(get_current_python_executable()),
        str(RUN_SCRIPT),
        mode,
        "--experiment-variant", variant.variant_id,
        "--run-id", run_id,
        "--output-root", str(output_root),
        "--device", "auto",
    ]
    if evaluate_only:
        command.extend(
            [
                "--evaluate-only",
                "--checkpoint",
                "best",
                "--windows-safe-mode",
                "--num-workers",
                "0",
                "--diagnostics-level",
                "minimal",
                "--prediction-accumulation",
                "streaming",
            ]
        )
    elif resume:
        command.append("--resume")
    if args.extra_args:
        command.extend(args.extra_args)
    return command


def _resume_plan(
    args: argparse.Namespace,
    existing_audit: dict[str, Any] | None,
    variant_dir: Path,
    variant_id: str,
) -> tuple[bool, str]:
    """Resolve whether a formal variant may reuse its last checkpoint.

    A missing checkpoint is treated as a fresh start so a single continuation
    command can cover both unfinished and not-yet-started variants. Existing
    artifacts with a mismatched protocol are different: silently passing them
    to ``--resume`` could mix an obsolete ablation definition into the current
    run, so the caller must start that variant explicitly without ``--resume``.
    """

    if not args.resume:
        return False, "not_requested"
    if existing_audit is None:
        return False, "no_existing_artifacts"
    if not existing_audit.get("config_match") or not existing_audit.get("protocol_consistent"):
        differences = ", ".join(existing_audit.get("config_differences") or []) or "protocol audit failed"
        raise ValueError(
            f"{variant_id}: refusing --resume because existing artifacts do not match "
            f"the current formal definition ({differences}). Run this variant fresh "
            "without --resume before attempting continuation."
        )
    if not (variant_dir / "last_checkpoint.pt").is_file():
        return False, "missing_last_checkpoint"
    return True, "last_checkpoint"


def _metrics_files(directory: Path) -> list[Path]:
    direct = sorted(directory.glob("metrics_eval_h*.json"))
    if direct:
        return direct
    matches = sorted(directory.glob("*/metrics_eval_h*.json"))
    return matches


def _variant_run_dir(root: Path, variant, expected_config: STMGPromptConfig) -> Path:
    """Resolve an existing child directory without renaming or recreating it."""

    variant_root = root / variant.variant_id
    preferred = variant_root / expected_config.model_name
    if preferred.exists():
        return preferred
    candidates = sorted(item for item in variant_root.iterdir() if item.is_dir()) if variant_root.exists() else []
    for candidate in candidates:
        if (candidate / "active_config.json").exists() or (candidate / "config.json").exists():
            return candidate
    return preferred


def _write_precision_status_files(root: Path, manifest: dict[str, Any], statuses: list[dict[str, Any]]) -> None:
    if manifest.get("experiment_family") != "precision":
        return
    by_variant = {item["variant"]: item for item in statuses}
    failed = [
        item["variant"]
        for item in statuses
        if not is_completed_status(item.get("final_status"))
        and item.get("status") not in {"REFERENCE_ONLY", "DRY_RUN"}
    ]
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": manifest.get("run_id"),
        "variants": by_variant,
        "failed_variants": failed,
        "pending_variants": [item["variant"] for item in statuses if item.get("final_status") == "pending"],
    }
    write_json(root / "precision_ablation_status.json", payload)
    write_json(root / "precision_ablation_manifest.json", {**manifest, "completion": payload})
    with (root / "precision_ablation_val_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["variant", "status", "final_status", "training_complete", "evaluation_complete", "best_epoch", "best_val_score_h10"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in statuses:
            writer.writerow({field: item.get(field, "") for field in fields})
    write_json(
        root / "precision_ablation_failure_summary.json",
        {
            "updated_at": payload["updated_at"],
            "failed_variants": failed,
            "failures": [item for item in statuses if item.get("variant") in failed],
        },
    )


def _historical_returncode(root: Path, variant_id: str) -> int:
    path = root / "experiment_status.json"
    if not path.exists():
        return 0
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    entries = value if isinstance(value, list) else value.get("variants", []) if isinstance(value, dict) else []
    for item in entries:
        if item.get("variant") == variant_id and item.get("returncode") is not None:
            try:
                return int(item["returncode"])
            except (TypeError, ValueError):
                return 0
    return 0


def _load_statuses(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "experiment_status.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = value if isinstance(value, list) else value.get("variants", []) if isinstance(value, dict) else []
    return {
        str(item["variant"]).upper(): item
        for item in entries
        if isinstance(item, dict) and item.get("variant")
    }


def _ordered_statuses(mapping, by_variant: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [by_variant[name] for name in mapping if name in by_variant]


def _current_variant_metrics(root: Path, variant) -> list[Path]:
    if variant.variant_id not in {"A4", "A5", "A6", "A7", "A9"}:
        return _metrics_files(root / variant.variant_id)
    expected_config = apply_variant(STMGPromptConfig(), variant.variant_id, variant.experiment_family)
    run_dir = _variant_run_dir(root, variant, expected_config)
    config_path = run_dir / "active_config.json"
    if not config_path.exists():
        config_path = run_dir / "config.json"
    try:
        actual = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if "st_prompt_mode" not in actual:
        actual["st_prompt_mode"] = "full"
    report = config_diff(actual, variant.variant_id, variant.experiment_family)
    return _metrics_files(run_dir) if report["passed"] else []


def _summary_rows(root: Path, variants) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical = canonical_directory(PROJECT_ROOT)
    for variant in variants:
        files = _metrics_files(canonical) if not variant.trainable else _current_variant_metrics(root, variant)
        for path in files:
            metric = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "variant": variant.variant_id,
                "display_name": variant.display_name,
                "trainable": variant.trainable,
                "canonical_reference": variant.canonical_reference or "",
                "metrics_source": str(path.resolve()),
                **metric,
            }
            rows.append(row)
    return rows


def write_summary(root: Path, variants) -> None:
    rows = _summary_rows(root, variants)
    write_json(root / "metrics_summary.json", rows)
    path = root / "metrics_summary.csv"
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_family(family: str, argv: list[str] | None = None) -> dict[str, Any]:
    args = parser_for(family).parse_args(argv)
    _validate_python_override(args.python)
    if sum(bool(value) for value in (args.run_full, args.run_smoke, args.full_shape_smoke, args.dry_run)) > 1:
        raise ValueError("Choose only one of --run-full, --run-smoke, --full-shape-smoke, or --dry-run.")
    mapping, _, _, _ = _family_data(family)
    variants = selected_variants(family, args.variants)
    all_variants = list(mapping.values())
    selected_ids = {item.variant_id for item in variants}
    root = _root(args, family)
    root.mkdir(parents=True, exist_ok=True)
    write_variant_matrix(root / "variant_config_matrix.csv", all_variants)
    manifest = {
        "canonical_id": CANONICAL_ID,
        "experiment_family": family,
        "run_id": root.name,
        "selection_uses_test_metrics": False,
        "variants": [item.to_dict() for item in all_variants],
        "selected_variants": [item.variant_id for item in variants],
    }
    write_json(root / "experiment_manifest.json", manifest)
    statuses_by_variant = _load_statuses(root)
    statuses_by_variant = {name: value for name, value in statuses_by_variant.items() if name in mapping}
    for name in selected_ids:
        statuses_by_variant.pop(name, None)

    def record_status(value: dict[str, Any]) -> list[dict[str, Any]]:
        statuses_by_variant[value["variant"]] = value
        return _ordered_statuses(mapping, statuses_by_variant)

    execute = args.run_full or args.run_smoke or args.full_shape_smoke
    for variant in variants:
        effective_dict, _ = _effective_artifacts(root, variant)
        expected_config = apply_variant(STMGPromptConfig(), variant.variant_id, variant.experiment_family)
        expected_config.output_root = str(root.parent.resolve())
        expected_config.run_id = f"{root.name}/{variant.variant_id}"
        variant_dir = _variant_run_dir(root, variant, expected_config)
        if not variant.trainable:
            print(f"{variant.variant_id}: REFERENCE_ONLY", flush=True)
            write_reference(variant.variant_id, root / variant.variant_id, PROJECT_ROOT)
            statuses = record_status(
                {
                    "variant": variant.variant_id,
                    "status": "REFERENCE_ONLY",
                    "final_status": "completed",
                    "returncode": 0,
                    "decision": "REFERENCE_ONLY",
                }
            )
            _write_precision_status_files(root, manifest, statuses)
            continue

        existing_audit = inspect_variant_artifacts(
            variant_dir,
            variant.variant_id,
            expected_config,
            returncode=_historical_returncode(root, variant.variant_id),
        ) if variant_dir.exists() else None
        if args.skip_completed and existing_audit and is_completed_status(existing_audit["final_status"]):
            print(f"{variant.variant_id}: SKIP_COMPLETED", flush=True)
            statuses = record_status(
                {
                    **existing_audit,
                    "status": existing_audit["final_status"],
                    "decision": "SKIP_COMPLETED",
                    "returncode": _historical_returncode(root, variant.variant_id),
                }
            )
            _write_precision_status_files(root, manifest, statuses)
            continue

        if not execute:
            statuses = record_status(
                {
                    "variant": variant.variant_id,
                    "status": "DRY_RUN",
                    "final_status": existing_audit["final_status"] if existing_audit else "pending",
                    "returncode": None,
                    "decision": "DRY_RUN",
                    **({"artifact_audit": existing_audit} if existing_audit else {}),
                }
            )
            _write_precision_status_files(root, manifest, statuses)
            continue

        evaluate_only = bool(
            existing_audit
            and existing_audit["training_complete"]
            and not existing_audit["evaluation_complete"]
            and args.resume
        )
        resume_used, resume_reason = _resume_plan(
            args,
            existing_audit,
            variant_dir,
            variant.variant_id,
        )
        if evaluate_only:
            resume_used = False
            resume_reason = "evaluate_only_best_checkpoint"
        action = "EVALUATE_ONLY" if evaluate_only else ("RESUME" if resume_used else "TRAIN")
        print(f"{variant.variant_id}: {action}", flush=True)
        command = _command(
            args,
            family,
            root,
            variant,
            evaluate_only=evaluate_only,
            resume=resume_used,
        )
        write_json(root / variant.variant_id / "run_command.json", command)
        command = normalize_python_command(command)
        print(
            f"Parent Python: {get_current_python_executable()}",
            flush=True,
        )
        print(f"Child Python: {command[0]}", flush=True)
        print(
            "Child command: " + subprocess.list2cmdline(command),
            flush=True,
        )
        start_time = datetime.now().isoformat(timespec="seconds")
        child = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=os.environ.copy())
        pid = child.pid
        returncode = child.wait()
        end_time = datetime.now().isoformat(timespec="seconds")
        final_audit = inspect_variant_artifacts(
            _variant_run_dir(root, variant, expected_config),
            variant.variant_id,
            expected_config,
            returncode=returncode,
        )
        status = final_audit["final_status"]
        if args.run_smoke and status == "completed":
            status = "smoke_completed"
        elif args.full_shape_smoke and status == "completed":
            status = "full_shape_smoke_completed"
        variant_status = {
            **final_audit,
            "status": status,
            "command": command,
            "pid": pid,
            "start_time": start_time,
            "end_time": end_time,
            "decision": action,
            "resume_requested": bool(args.resume),
            "resume_used": resume_used,
            "resume_reason": resume_reason,
        }
        statuses = record_status(variant_status)
        write_json(root / "experiment_status.json", statuses)
        _write_precision_status_files(root, manifest, statuses)
        if returncode and not is_completed_status(final_audit["final_status"]) and not args.continue_on_error:
            break
    statuses = _ordered_statuses(mapping, statuses_by_variant)
    write_json(root / "experiment_status.json", statuses)
    _write_precision_status_files(root, manifest, statuses)
    write_summary(root, all_variants)
    failed = [
        item["variant"]
        for item in statuses
        if item.get("variant") in selected_ids
        and item.get("variant") not in {"P0", "A0"}
        and item.get("status") not in {"DRY_RUN", "REFERENCE_ONLY"}
        and not is_completed_status(item.get("final_status"))
    ]
    result = {
        "root": str(root.resolve()),
        "family": family,
        "statuses": statuses,
        "failed_variants": failed,
        "formal_training_started": bool(args.run_full),
        "smoke_started": bool(args.run_smoke or args.full_shape_smoke),
    }
    if failed:
        raise SystemExit(f"Formal variants failed: {failed}")
    return result
