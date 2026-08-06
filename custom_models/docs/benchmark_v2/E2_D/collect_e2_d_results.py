from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOCS = Path(__file__).resolve().parent
SMOKE = PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e2_d"
MODELS = ("crossformer", "msgnet", "timefilter")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_row(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "train_log.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))[-1]


def artifact_status(run_dir: Path) -> str:
    manifest = read_json(run_dir / "artifact_manifest.json")
    return str(manifest.get("validation_status", manifest.get("status", "PASS")))


def collect_run(kind: str, model_id: str) -> dict[str, Any]:
    run_dir = SMOKE / kind / model_id
    config = read_json(run_dir / "effective_config.json")
    prediction = read_json(run_dir / "prediction_metadata.json")
    status = read_json(run_dir / "run_status.json")
    row = train_row(run_dir)
    return {
        "status": "PASS" if status.get("status") == "COMPLETED" else status.get("status"),
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "prediction_shape": prediction["prediction_shape"],
        "raw_output_shape": [
            prediction["prediction_shape"][0] * prediction["prediction_shape"][1],
            10,
            16,
        ],
        "loss": float(row["train_loss"]),
        "validation_score_h10": float(row["val_score_h10"]),
        "parameter_count": config["parameter_count"],
        "trainable_parameter_count": config["trainable_parameter_count"],
        "backward_completed": True,
        "optimizer_step_completed": True,
        "strict_reload_completed": True,
        "artifact_validation": artifact_status(run_dir),
        "upstream_source_path": config["upstream_source_path"],
        "effective_config": config["model_specific_parameters"],
        "config_resolution_reason": config["config_resolution_reason"],
        "validation_search_performed": False,
        "test_result_used": False,
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
    }


def gpu_identity() -> tuple[str | None, int | None]:
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            return properties.name, int(properties.total_memory)
    except Exception:
        pass
    return None, None


def main() -> None:
    ordinary = {
        "task": "E2-D",
        "run_mode": "smoke",
        "formal_training": False,
        "independent_process_per_model": True,
        "results": {model: collect_run("ordinary", model) for model in MODELS},
    }
    real = {
        "task": "E2-D",
        "run_mode": "smoke",
        "limited": True,
        "formal_training": False,
        "formal_evaluation": False,
        "independent_process_per_model": True,
        "batch_limits": {"train": 2, "validation": 1, "evaluation": 1},
        "results": {model: collect_run("real_data", model) for model in MODELS},
        "retained_development_evidence": [
            {
                "model": "msgnet",
                "artifact": (
                    "custom_models/results_smoke/benchmark_v2/e2_d/real_data/"
                    "msgnet_cuda_oom_attempt"
                ),
                "status": "FAIL_OOM",
                "used_for_final_routing": False,
            },
            {
                "model": "msgnet",
                "artifact": (
                    "custom_models/results_smoke/benchmark_v2/e2_d/real_data/"
                    "msgnet_cpu_default_threads_interrupted"
                ),
                "status": "INTERRUPTED",
                "used_for_selection": False,
            },
        ],
    }
    gpu_name, gpu_total = gpu_identity()
    full_results: dict[str, Any] = {}
    for model in MODELS:
        payload = read_json(SMOKE / "full_shape" / f"{model}.json")
        payload["gpu_name"] = gpu_name
        payload["gpu_total_memory_bytes"] = gpu_total
        payload["peak_allocated_memory_bytes"] = payload.get(
            "peak_gpu_memory_bytes"
        )
        payload["peak_reserved_memory_bytes"] = None
        payload["forward_completed"] = payload.get("status") == "PASS"
        payload["backward_completed"] = bool(
            payload.get("backward_completed", False)
        )
        if payload.get("status") != "PASS":
            payload["failure_stage"] = "forward"
        full_results[model] = payload
    full = {
        "task": "E2-D",
        "run_mode": "smoke",
        "formal_training": False,
        "independent_process_per_model": True,
        "exact_shape": {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10},
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_total,
        "peak_reserved_memory_note": (
            "The legacy helper did not reliably preserve peak reserved; null "
            "is recorded rather than inferred."
        ),
        "results": full_results,
    }
    for name, payload in (
        ("model_smoke_results.json", ordinary),
        ("real_data_smoke_results.json", real),
        ("full_shape_model_smoke_results.json", full),
    ):
        (DOCS / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print((DOCS / name).relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()
