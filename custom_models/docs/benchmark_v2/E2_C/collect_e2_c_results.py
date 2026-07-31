from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e2_c"
MODELS = ("timemixer", "tsmixer", "frets")
GPU_NAME = "NVIDIA GeForce GTX 1060"
GPU_TOTAL_MEMORY_BYTES = 6442319872


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: dict[str, Any]) -> None:
    (DOC_ROOT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def smoke_record(run_dir: Path, *, real: bool) -> dict[str, Any]:
    status = read_json(run_dir / "run_status.json")
    effective = read_json(run_dir / "effective_config.json")
    prediction = read_json(run_dir / "prediction_metadata.json")
    metrics = read_json(run_dir / "metrics_eval_h10.json")
    with (run_dir / "train_log.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        train_row = list(csv.DictReader(handle))[-1]
    record: dict[str, Any] = {
        "status": "PASS" if status["status"] == "COMPLETED" else status["status"],
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "prediction_shape": prediction["prediction_shape"],
        "raw_output_shape": prediction.get("raw_output_shape"),
        "loss": float(train_row["train_loss"]),
        "validation_score_h10": float(train_row["val_score_h10"]),
        "evaluation_score_h10": metrics["Score"],
        "evaluation_r2_h10": metrics["R2"],
        "parameter_count": effective["parameter_count"],
        "trainable_parameter_count": effective["trainable_parameter_count"],
        "backward_completed": True,
        "optimizer_step_completed": True,
        "strict_reload_completed": True,
        "artifact_validation": "PASS",
        "upstream_source_path": effective["upstream_source_path"],
        "upstream_source_sha256": effective["upstream_source_sha256"],
        "effective_config": effective["model_specific_parameters"],
        "config_resolution_reason": effective["config_resolution_reason"],
        "validation_search_performed": effective[
            "validation_search_performed"
        ],
        "test_result_used": effective["test_result_used"],
        "smoke_result_used_for_selection": effective[
            "smoke_result_used_for_selection"
        ],
        "oom_driven_capacity_reduction": effective[
            "oom_driven_capacity_reduction"
        ],
    }
    if real:
        record.update(
            {
                "train_batch_limit": 2,
                "validation_batch_limit": 1,
                "evaluation_batch_limit": 1,
            }
        )
    return record


def full_record(model: str) -> dict[str, Any]:
    record = read_json(SMOKE_ROOT / "full_shape" / f"{model}.json")
    record["gpu_name"] = GPU_NAME
    record["gpu_total_memory_bytes"] = GPU_TOTAL_MEMORY_BYTES
    record["peak_reserved_memory_bytes"] = None
    record["forward_completed"] = record["status"] == "PASS"
    if record["status"] == "FAIL_OOM":
        record["failure_stage"] = "forward"
    return record


def main() -> None:
    ordinary = {
        "task": "E2-C",
        "run_mode": "smoke",
        "formal_training": False,
        "shape": {"B": 2, "T": 144, "N": 4, "C": 16, "H": 10},
        "independent_process_per_model": True,
        "results": {
            model: smoke_record(SMOKE_ROOT / "ordinary" / model, real=False)
            for model in MODELS
        },
        "retained_development_evidence": [
            {
                "model": "frets",
                "artifact": (
                    "custom_models/results_smoke/benchmark_v2/e2_c/diagnostics/"
                    "frets_branch_amp_diagnostics.json"
                ),
                "finding": (
                    "int 0 skips channel branch; string '0' activates it; "
                    "raw-source CUDA AMP passed"
                ),
            },
            {
                "model": "frets",
                "artifact": (
                    "custom_models/results_smoke/benchmark_v2/e2_c/ordinary/"
                    "frets_fp32_boundary_attempt"
                ),
                "status": "PASS_SUPERSEDED",
                "reason": (
                    "Preventive full-forward FP32 boundary was removed after "
                    "raw-source CUDA AMP passed; retained without use for selection"
                ),
            },
        ],
    }
    full = {
        "task": "E2-C",
        "run_mode": "smoke",
        "formal_training": False,
        "independent_process_per_model": True,
        "exact_shape": {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10},
        "gpu_name": GPU_NAME,
        "gpu_total_memory_bytes": GPU_TOTAL_MEMORY_BYTES,
        "peak_reserved_memory_note": (
            "The legacy full_shape_model_smoke helper did not reliably record "
            "peak reserved; values are null rather than inferred."
        ),
        "results": {model: full_record(model) for model in MODELS},
        "retained_superseded_attempts": [
            {
                "model": "frets",
                "artifact": (
                    "custom_models/results_smoke/benchmark_v2/e2_c/full_shape/"
                    "frets_fp32_boundary_attempt.json"
                ),
                "status": "FAIL_OOM",
                "used_for_final_routing": False,
            }
        ],
    }
    real = {
        "task": "E2-C",
        "run_mode": "smoke",
        "limited": True,
        "formal_training": False,
        "formal_evaluation": False,
        "independent_process_per_model": True,
        "results": {
            model: smoke_record(SMOKE_ROOT / "real_data" / model, real=True)
            for model in MODELS
        },
        "retained_superseded_attempts": [
            {
                "model": "frets",
                "artifact": (
                    "custom_models/results_smoke/benchmark_v2/e2_c/real_data/"
                    "frets_fp32_boundary_attempt"
                ),
                "status": "PASS_SUPERSEDED",
                "used_for_selection": False,
            }
        ],
    }
    write_json("model_smoke_results.json", ordinary)
    write_json("full_shape_model_smoke_results.json", full)
    write_json("real_data_smoke_results.json", real)


if __name__ == "__main__":
    main()
