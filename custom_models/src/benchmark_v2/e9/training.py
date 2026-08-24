from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark_v2.artifacts import atomic_write_json, safe_run_dir, validate_run, write_status
from benchmark_v2.data import SDWPFDataProvider
from benchmark_v2.engine import Evaluator, Trainer
from benchmark_v2.losses import build_msmg_dwu_loss
from benchmark_v2.model_cli import DEFAULT_INPUT_PATH, DEFAULT_TARGET_PATH, _write_common, _write_metrics
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.precision import apply_model_precision_policy
from benchmark_v2.protocol import load_protocol
from benchmark_v2.runtime import ProviderBatchIterable
from benchmark_v2.seeds import seed_everything
from benchmark_v2.training_profiles import apply_training_profile, resolved_batch_sizes

from .config_diff import build_transfer_config, compare_configs
from .constants import LOSS_PROFILE, MODEL_IDS, PREFLIGHT_ROOT, TRAINING_PROFILE_ID, TRANSFER_ROOT
from .io_utils import read_json
from .variants import CONTROL_RUN_IDS, canonical_transfer_run_id


def _control_dir(model_id: str) -> Path:
    from .constants import PROJECT_ROOT, SCOPE_MANIFEST
    scope = read_json(SCOPE_MANIFEST)
    entry = next(row for row in scope["entries"] if row.get("model_id") == model_id)
    if entry.get("run_id") != CONTROL_RUN_IDS[model_id]:
        raise RuntimeError("Frozen control run-id conflicts with E9 variant manifest.")
    return (PROJECT_ROOT / entry["output_root"] / entry["run_id"]).resolve()


def _preflight_pass(model_id: str, preflight_root: str | Path) -> bool:
    path = Path(preflight_root) / "E9_PREFLIGHT_INVENTORY.json"
    if not path.is_file():
        return False
    payload = read_json(path)
    matches = [row for row in payload.get("inventory", []) if row.get("model_id") == model_id]
    return len(matches) == 1 and matches[0].get("status") == "PASS" and matches[0].get("run_id") == canonical_transfer_run_id(model_id)


def _current_runtime_compatible(current: dict[str, Any], control: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    ignored = {
        "run_id", "output_root", "scope_id", "entry_id", "active_scope_id", "current_scope_entry_id",
        "current_scope_manifest_hash", "formal_worker_pid", "source_revision", "provenance",
        "upstream_source_path", "upstream_license_path",
    }
    mismatches = []
    for key, value in current.items():
        if key in ignored or key not in control:
            continue
        if control[key] != value:
            mismatches.append({"path": key, "control": control[key], "current": value})
    return not mismatches, mismatches


def _existing_completed(run_dir: Path, model_id: str) -> dict[str, Any] | None:
    if not run_dir.exists():
        return None
    effective_path, status_path = run_dir / "effective_config.json", run_dir / "run_status.json"
    if effective_path.is_file() and status_path.is_file():
        effective, status = read_json(effective_path), read_json(status_path)
        if (
            status.get("status") == "COMPLETED" and effective.get("model_id") == model_id
            and effective.get("loss") == "msmg_dwu_loss" and effective.get("run_id") == run_dir.name
        ):
            validate_run(run_dir, expected_training_batch_profile_id=TRAINING_PROFILE_ID)
            return {"status": "SKIPPED_COMPLETED_IDENTITY_MATCH", "model_id": model_id, "run_dir": str(run_dir)}
    raise RuntimeError(f"Existing E9 run directory is incomplete or identity-conflicting and will not be overwritten: {run_dir}")


def formal_train_transfer(
    model_id: str, *, input_path: str | Path = DEFAULT_INPUT_PATH, target_path: str | Path = DEFAULT_TARGET_PATH,
    output_root: str | Path = TRANSFER_ROOT, device: str = "cuda", preflight_root: str | Path = PREFLIGHT_ROOT,
) -> dict[str, Any]:
    if model_id not in MODEL_IDS:
        raise ValueError(f"Model is outside the frozen E9 scope: {model_id}")
    run_id = canonical_transfer_run_id(model_id)
    run_dir = safe_run_dir(output_root, run_id)
    existing = _existing_completed(run_dir, model_id)
    if existing is not None:
        return existing
    if not _preflight_pass(model_id, preflight_root):
        raise RuntimeError(f"Exact E9 MS-MG-DWU preflight PASS is required before formal training: {model_id}")
    protocol = load_protocol()
    seed_everything(int(protocol["default_seed"]))
    provider = SDWPFDataProvider.from_files(input_path, target_path, protocol=protocol)
    runtime = build_model_runtime(model_id, protocol, run_mode="formal", target_scaler=provider.scalers["target"])
    apply_training_profile(runtime, TRAINING_PROFILE_ID)
    apply_model_precision_policy(runtime)
    control_dir = _control_dir(model_id)
    control_effective = read_json(control_dir / "effective_config.json")
    compatible, mismatches = _current_runtime_compatible(runtime.effective_config, control_effective)
    if not compatible:
        raise RuntimeError(f"Current model resolver differs from the frozen control: {mismatches}")
    effective_transfer = build_transfer_config(control_effective, model_id, output_root)
    effective_transfer.pop("formal_worker_pid", None)
    effective_transfer["formal_worker_pid"] = os.getpid()
    runtime.effective_config = deepcopy(effective_transfer)
    diff = compare_configs(control_effective, effective_transfer, graph_model=model_id in {"dcrnn", "mtgnn"})
    if not diff["loss_only_diff_valid"]:
        raise RuntimeError(f"E9 loss-only config diff failed closed: {diff['unexpected_differences']}")
    run_dir.mkdir(parents=True, exist_ok=False)
    loss_fn = build_msmg_dwu_loss(**LOSS_PROFILE)
    atomic_write_json(run_dir / "initialization_provenance.json", {
        "initialization_seed": int(protocol["default_seed"]),
        "initial_model_state_hash": None,
        "exact_initial_state_pairing": "NOT_VERIFIED",
        "rng_manifest": {"seed": int(protocol["default_seed"]), "torch_rng_saved_in_checkpoint": True},
        "sampler_identity": "ProviderBatchIterable sequential formal starts",
        "data_order_identity": "strict chronological train starts, stride=6, no shuffle",
    })
    atomic_write_json(run_dir / "control_reference.json", {
        "source_class": "ORIGINAL26_CONTROL", "read_only": True,
        "model_id": model_id, "run_id": CONTROL_RUN_IDS[model_id], "source_root": str(control_dir),
    })
    atomic_write_json(run_dir / "loss_only_config_diff.json", diff)
    atomic_write_json(run_dir / "loss_state_manifest.json", {
        "fit_split": "train", "validation_test_frozen": True, "uses_test_target": False,
        "model_output_dependent": True, "algorithm_shared": True, "values_shared": False,
        "checkpointed": True,
    })
    resolved, effective = _write_common(
        run_dir, runtime, protocol, profile="TRAIN", run_mode="formal",
        data_signature=provider.signature(), formal_training=True,
    )
    trainer = Trainer(
        model=runtime.model, adapter=runtime.adapter, loss_fn=loss_fn, protocol=protocol,
        run_dir=run_dir, model_id=model_id, resolved_config=resolved, effective_config=effective,
        device=device, inverse_target=runtime.inverse_target,
    )
    try:
        sizes = resolved_batch_sizes(protocol, TRAINING_PROFILE_ID)
        history = trainer.fit(
            ProviderBatchIterable(provider, "train", sizes["train"]),
            ProviderBatchIterable(provider, "val", sizes["val"]),
        )
        metrics = Evaluator(trainer).evaluate(ProviderBatchIterable(provider, "test", sizes["test"]))
        _write_metrics(
            run_dir, metrics, runtime=runtime, protocol=protocol,
            prediction_shape=[len(provider.starts["test"]), len(provider.node_ids), int(protocol["max_pred_len"])],
            source_checkpoint="best_checkpoint.pt",
        )
        write_status(run_dir, status="COMPLETED", run_mode="formal", artifact_profile="TRAIN", formal_training=True, exit_code=0)
        return {
            "status": "PASS", "model_id": model_id, "run_dir": str(run_dir),
            "epochs_completed": len(history), "metrics": metrics,
            "artifact_validation": validate_run(run_dir, expected_training_batch_profile_id=TRAINING_PROFILE_ID),
        }
    except Exception:
        if not (run_dir / "run_status.json").is_file():
            write_status(run_dir, status="FAILED", run_mode="formal", artifact_profile="FAILED", formal_training=True, exit_code=1)
        raise
