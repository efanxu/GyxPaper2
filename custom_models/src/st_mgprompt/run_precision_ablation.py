from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from st_mgprompt.artifact_status import inspect_variant_artifacts
from st_mgprompt.formal_runner import run_family


def _safe_optional_float(value):
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _should_train_variant(args, artifact: dict) -> bool:
    """Compatibility helper used by regression tests and older notebooks."""

    if getattr(args, "force_rerun", False):
        return True
    if getattr(args, "skip_completed", False) and artifact.get("completed"):
        return False
    return not bool(artifact.get("completed"))


def read_best_val_summary(root: Path, manifest: list[dict], status: dict | None = None) -> list[dict]:
    status = status or {}
    rows = []
    for item in manifest:
        variant = item.get("variant") or item.get("variant_id")
        current = status.get(variant, {})
        run_dir = root / variant / str(item.get("model_name", ""))
        checkpoint_values: dict = {}
        checkpoint_path = run_dir / "best_checkpoint.pt"
        if checkpoint_path.exists():
            try:
                import torch

                checkpoint_values = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception:
                checkpoint_values = {}
        train_values: dict = {}
        train_path = run_dir / "train_complete.json"
        if train_path.exists():
            try:
                train_values = json.loads(train_path.read_text(encoding="utf-8"))
            except Exception:
                train_values = {}
        model_summary: dict = {}
        summary_path = run_dir / "model_summary.json"
        if summary_path.exists():
            try:
                model_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                model_summary = {}

        def resolved(name: str):
            return current.get(name, checkpoint_values.get(name, train_values.get(name)))

        value = resolved("best_val_score_h10")
        rows.append(
            {
                "variant": variant,
                "status": current.get("status", "pending"),
                "best_epoch": resolved("best_epoch"),
                "best_val_Score_H3": resolved("best_val_score_h3"),
                "best_val_Score_H6": resolved("best_val_score_h6"),
                "best_val_Score_H10": value,
                "trainable_parameters": model_summary.get("trainable_parameters"),
                "relative_improvement_vs_P0_H10": "",
            }
        )
    return rows


def _mark_dependents_blocked_by_failed_baseline(root: Path, manifest: list[dict], status: dict) -> None:
    for item in manifest:
        variant = item.get("variant") or item.get("variant_id")
        if variant != "P0":
            status.setdefault(variant, {}).update({"status": "blocked_by_failed_baseline"})
    (root / "precision_ablation_failure_summary.json").write_text(
        json.dumps({"reason": "P0 failed", "status": status}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    try:
        result = run_family("precision")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
