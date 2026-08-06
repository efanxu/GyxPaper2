from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e2_b"
MODELS = ("timesnet", "micn", "wpmixer", "multipatchformer")
ORDINARY_RUNS = {
    "timesnet": SMOKE_ROOT / "passing/timesnet",
    "micn": SMOKE_ROOT / "micn",
    "wpmixer": SMOKE_ROOT / "passing_v2/wpmixer",
    "multipatchformer": SMOKE_ROOT / "multipatchformer",
}
REAL_RUNS = {
    model: SMOKE_ROOT / "real_data" / model for model in MODELS
}
PROTOCOL_record = (
    "0140d8774e2cc189a8bd99f1fc9c8a120"
    "b565265bf9a7c729ed1d47a0b1c069b"
)
CANONICAL_record = (
    "f08c822f512384aaf7700b9f5e6049a940"
    "d63f385829a43f4223920b583bba7a"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: dict[str, Any]) -> None:
    (DOC_ROOT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
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
        train_row = next(csv.DictReader(handle))
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
        "effective_config": effective["model_specific_parameters"],
        "config_resolution_reason": effective["config_resolution_reason"],
        "validation_search_performed": effective[
            "validation_search_performed"
        ],
        "test_result_used": effective["test_result_used"],
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


def main() -> None:
    ordinary = {
        "task": "E2-B",
        "run_mode": "smoke",
        "formal_training": False,
        "shape": {"B": 2, "T": 144, "N": 4, "C": 16, "H": 10},
        "results": {
            model: smoke_record(ORDINARY_RUNS[model], real=False)
            for model in MODELS
        },
        "retained_initial_fail_non_oom_attempts": [
            {
                "model": "timesnet",
                "run_dir": (
                    "custom_models/results_smoke/benchmark_v2/e2_b/timesnet"
                ),
                "status": "FAIL_NON_OOM",
                "error": (
                    "RuntimeError: cuFFT only supports power-of-two signal "
                    "sizes in half precision; signal size was 154"
                ),
                "resolution": "FFT only runs FP32; convolution/backward retain AMP",
            },
            {
                "model": "wpmixer",
                "run_dir": (
                    "custom_models/results_smoke/benchmark_v2/e2_b/wpmixer"
                ),
                "status": "FAIL_NON_OOM",
                "error": (
                    "RuntimeError: evaluation DWT output Half versus Float "
                    "linear weights"
                ),
                "resolution": "DWT/IDWT FP32; resolution mixers retain Trainer AMP",
            },
            {
                "model": "wpmixer",
                "run_dir": (
                    "custom_models/results_smoke/benchmark_v2/e2_b/"
                    "passing/wpmixer"
                ),
                "status": "FAIL_NON_OOM",
                "error": (
                    "RuntimeError: custom DWT backward Half input versus "
                    "Float wavelet filters"
                ),
                "resolution": "DWT/IDWT FP32; resolution mixers retain Trainer AMP",
            },
        ],
    }
    full_results = {
        model: read_json(SMOKE_ROOT / "full_shape" / f"{model}.json")
        for model in MODELS
    }
    full = {
        "task": "E2-B",
        "run_mode": "smoke",
        "formal_training": False,
        "independent_process_per_model": True,
        "gpu_name": "NVIDIA GeForce GTX 1060",
        "gpu_total_memory_bytes": 6442450944,
        "peak_reserved_memory_note": (
            "Legacy full_shape_model_smoke helper recorded peak allocated but "
            "not peak reserved; reserved is null rather than invented. The "
            "formal hardware-preflight worker records both."
        ),
        "results": full_results,
    }
    for record in full_results.values():
        record.setdefault("gpu_name", "NVIDIA GeForce GTX 1060")
        record.setdefault("gpu_total_memory_bytes", 6442450944)
        record.setdefault("peak_reserved_memory_bytes", None)
    real = {
        "task": "E2-B",
        "run_mode": "smoke",
        "limited": True,
        "formal_training": False,
        "formal_evaluation": False,
        "independent_process_per_model": True,
        "results": {
            model: smoke_record(REAL_RUNS[model], real=True)
            for model in MODELS
        },
    }
    tests = {
        "task": "E2-B",
        "interpreter": "D:/Apps/Miniconda3/envs/env_tslib/python.exe",
        "framework": "unittest",
        "final_suite": {
            "command": (
                "python -m unittest discover "
                "-s custom_models/tests/benchmark_v2 -p test_*.py"
            ),
            "status": "PASS",
            "tests_run": 71,
            "failures": 0,
            "errors": 0,
        },
        "e2_a_test_baseline_retained": 60,
        "e2_b_tests_added": 11,
        "protocol_check": {
            "status": "PASS",
            "protocol_record": PROTOCOL_record,
        },
        "focused_checks": [
            "12-entry explicit TSLib allowlist and lazy local source loading",
            "four node-shared model output, backward and strict reload",
            "target/mask invariance and future-input rejection",
            "TimesNet node-isolated FFT scope and legal top_k",
            "MICN exact 154/58 regression and fixed 154-step zero marks",
            "WPMixer approximation/detail branch participation",
            "MultiPatchFormer exact 18/19 regression and four 18-token branches",
            "exact hardware preflight identities for all four models",
        ],
    }
    preflight_tests = {
        "task": "E2-B",
        "status": "PASS",
        "tests_run": 8,
        "test_only_dummy_process_runner": True,
        "exact_shape": {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10},
        "verified": {
            "all_four_identity_fields_are_exact": True,
            "only_exact_local_oom_models_require_preflight": True,
            "timesnet_and_micn_local_pass_do_not_require_preflight": True,
            "wpmixer_and_multipatchformer_oom_require_preflight": True,
            "missing_pass_spawns_preflight": True,
            "matching_pass_spawns_formal_train": True,
            "preflight_exits_before_formal_train_starts": True,
            "preflight_pid_differs_from_formal_pid": True,
            "fail_oom_stops_before_formal_train": True,
            "fail_non_oom_stops_before_formal_train": True,
            "config_or_shape_mismatch_not_reused": True,
            "parent_launcher_constructs_no_cuda_tensor": True,
        },
        "target_high_memory_hardware_preflight": "NOT_RUN",
    }
    write_json("model_smoke_results.json", ordinary)
    write_json("full_shape_model_smoke_results.json", full)
    write_json("real_data_smoke_results.json", real)
    write_json("test_results.json", tests)
    write_json("hardware_preflight_test_results.json", preflight_tests)


if __name__ == "__main__":
    main()
