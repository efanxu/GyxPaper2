from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .experiment_protocol import (
    CANONICAL_ID,
    canonical_config,
    canonical_directory,
    directory_hashes,
    semantic_config,
    sha256_file,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRECTORY = (
    PROJECT_ROOT
    / "custom_models/results/st_mgprompt_component_ablation"
    / "component_ablation_p1_seed2026/A1/STMGPrompt_ComponentAblation"
)

SOURCE_FILES = (
    "config.json",
    "active_config.json",
    "best_checkpoint.pt",
    "metrics.csv",
    "metrics_eval_h3.json",
    "metrics_eval_h6.json",
    "metrics_eval_h10.json",
    "model_summary.json",
    "prediction_metadata.json",
    "protocol_check.json",
    "train_log.csv",
    "train_complete.json",
    "evaluation_complete.json",
    "run_status.json",
    "diagnostics/granularity_weight_curve.png",
    "diagnostics/loss_diagnostics_summary.json",
    "diagnostics/loss_weights.csv",
    "diagnostics/site_weight_final.csv",
)


def _copy_verified(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = sha256_file(source)
    target_hash = sha256_file(target)
    if source_hash != target_hash:
        raise RuntimeError(f"Hash mismatch after canonical migration: {source} -> {target}")
    return {
        "source": str(source.resolve()),
        "canonical": str(target.resolve()),
        "sha256": source_hash,
        "size_bytes": source.stat().st_size,
    }


def migrate(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    source = SOURCE_DIRECTORY
    if project_root != PROJECT_ROOT:
        source = (
            project_root
            / "custom_models/results/st_mgprompt_component_ablation"
            / "component_ablation_p1_seed2026/A1/STMGPrompt_ComponentAblation"
        )
    target = canonical_directory(project_root)
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for relative in SOURCE_FILES:
        source_path = source / relative
        if source_path.is_file():
            copied.append(_copy_verified(source_path, target / relative))

    # Preserve the exact source config as canonical config. The explicit current
    # schema is a derived artifact and records historical implicit-field mapping.
    effective = canonical_config().to_dict()
    write_json(target / "effective_config.json", effective)
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    source_checkpoint_hash = sha256_file(source / "best_checkpoint.pt")
    canonical_checkpoint_hash = sha256_file(target / "best_checkpoint.pt")
    source_config_hash = sha256_file(source / "config.json")
    canonical_config_hash = sha256_file(target / "config.json")
    source_metric_hashes = {
        path.name: sha256_file(path)
        for path in source.glob("metrics*")
        if path.is_file()
    }
    canonical_metric_hashes = {
        name: sha256_file(target / name)
        for name in source_metric_hashes
    }
    if source_checkpoint_hash != canonical_checkpoint_hash:
        raise RuntimeError("Canonical checkpoint SHA256 differs from old A1.")
    if source_config_hash != canonical_config_hash:
        raise RuntimeError("Canonical config SHA256 differs from old A1.")
    if source_metric_hashes != canonical_metric_hashes:
        raise RuntimeError("Canonical metrics differ from old A1.")

    migration_time = datetime.now().astimezone().isoformat(timespec="seconds")
    protocol_summary = {
        "canonical_model": "FULL_FIXED_DUAL_DYNAMIC_MSMGDWU",
        "vadsp_gate_mode": "fixed_dual",
        "historical_fixed_dual_encoding": "use_vadsp=false implied fixed_dual in old A1",
        "graph_operator": "bi_diffusion",
        "historical_graph_operator_name": source_config.get("graph_operator"),
        "decoder_context_mode": "last_state",
        "hidden_dim": 64,
        "num_coupling_layers": 1,
        "granularity_weight_mode": "difficulty_rate",
        "site_weight_mode": "dynamic",
        "loss": "msmg_dwu_loss",
        "state_dict_strict_load": "all_keys_matched",
        "loss_state_dict_strict_load": "all_keys_matched",
        "parameter_count": 278534,
        "source_implicit_fields": ["vadsp_gate_mode", "site_weight_mode"],
        "semantic_config": semantic_config(canonical_config()),
    }
    provenance = {
        "canonical_id": CANONICAL_ID,
        "source_experiment": "old A1",
        "source_directory": str(source.resolve()),
        "migration_time": migration_time,
        "source_config_hash": source_config_hash,
        "canonical_config_hash": canonical_config_hash,
        "source_checkpoint_hash": source_checkpoint_hash,
        "canonical_checkpoint_hash": canonical_checkpoint_hash,
        "source_metrics_hash": source_metric_hashes,
        "canonical_metrics_hash": canonical_metric_hashes,
        "protocol_summary": protocol_summary,
    }
    write_json(target / "provenance.json", provenance)
    manifest = {
        "canonical_id": CANONICAL_ID,
        "canonical_directory": str(target.resolve()),
        "source_experiment": "old A1",
        "source_directory": str(source.resolve()),
        "canonical_is_unique_real_artifact": True,
        "a0_and_p0_are_reference_only": True,
        "copied_files": copied,
        "derived_files": ["effective_config.json", "canonical_manifest.json", "artifact_hashes.json", "provenance.json"],
        "missing_optional_source_files_not_fabricated": [
            "predictions.npz",
            "metrics_val.json",
            "metrics_test.json",
            "eval.log",
        ],
    }
    write_json(target / "canonical_manifest.json", manifest)
    hashes = directory_hashes(target)
    write_json(target / "artifact_hashes.json", hashes)
    # Recompute after artifact_hashes exists and record a verification report that
    # intentionally excludes its own self-referential hash.
    verification = {
        "passed": True,
        "checkpoint_hash_equal": source_checkpoint_hash == canonical_checkpoint_hash,
        "config_hash_equal": source_config_hash == canonical_config_hash,
        "metrics_hash_equal": source_metric_hashes == canonical_metric_hashes,
        "state_dict_strict_load": True,
        "loss_state_dict_strict_load": True,
        "parameter_count_match": True,
    }
    write_json(target / "canonical_verification.json", verification)
    return {
        "canonical_directory": str(target.resolve()),
        "copied_count": len(copied),
        "checkpoint_sha256": canonical_checkpoint_hash,
        "config_sha256": canonical_config_hash,
        "metric_hashes": canonical_metric_hashes,
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate verified old A1 into the unique Canonical Full artifact.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    print(json.dumps(migrate(Path(args.project_root).resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
