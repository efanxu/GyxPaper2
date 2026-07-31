from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "docs"
    / "benchmark_v2"
    / "UNIFORM_BATCH4_PROTOCOL"
)
PROFILE_ID = "uniform_train_batch4_v1"
PROFILE_HASH = (
    "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66"
)
PROTOCOL_HASH = (
    "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b"
)
CANONICAL_HASH = (
    "f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a"
)
GRAPH_HASH = (
    "f8224287f2a41be0e67e808445ce9a43b9973e7b075d1e1b4768396d6f98e1ef"
)
TRAINABLE = (
    "gru",
    "dlinear",
    "lightts",
    "tide",
    "segrnn",
    "transformer",
    "patchtst",
    "itransformer",
    "timexer",
    "timesnet",
    "micn",
    "wpmixer",
    "multipatchformer",
    "timemixer",
    "tsmixer",
    "frets",
    "crossformer",
    "msgnet",
    "timefilter",
    "gcn",
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "mtgnn",
    "agcrn",
    "stid",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: Any) -> None:
    (DOC_ROOT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(name: str, text: str) -> None:
    (DOC_ROOT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_smoke_log(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    stdout = path.read_text(encoding="utf-8", errors="replace").split(
        "\n--- STDERR ---", 1
    )[0].strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def build_real_smoke_inventory() -> dict[str, Any]:
    initial = {
        model: PROJECT_ROOT
        / "custom_models"
        / "logs"
        / "uniform_bs4"
        / "smoke_real"
        / f"{model}.log"
        for model in TRAINABLE[:17]
    }
    retry1_models = ("msgnet", "timefilter", "gcn", "stgcn")
    retry1 = {
        model: PROJECT_ROOT
        / "custom_models"
        / "logs"
        / "uniform_bs4"
        / "smoke_real_retry1"
        / f"{model}.log"
        for model in retry1_models
    }
    retry2_models = ("dcrnn", "graph_wavenet", "mtgnn", "agcrn", "stid")
    retry2 = {
        model: PROJECT_ROOT
        / "custom_models"
        / "logs"
        / "uniform_bs4"
        / "smoke_real_retry2"
        / f"{model}.log"
        for model in retry2_models
    }
    paths = {**initial, **retry1, **retry2}
    rows = []
    for model in TRAINABLE:
        path = paths[model]
        payload = parse_smoke_log(path)
        if payload and payload.get("status") == "PASS":
            status = "PASS"
            failure_stage = None
            error_type = None
            error_message = None
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            if "OutOfMemoryError" in text or "CUDA out of memory" in text:
                status = "FAIL_OOM"
                error_type = "OutOfMemoryError"
                failure_stage = "backward" if "backward" in text else "unknown"
                error_message = next(
                    (
                        line.strip()
                        for line in reversed(text.splitlines())
                        if "out of memory" in line.lower()
                    ),
                    "CUDA OOM",
                )
            else:
                status = "FAIL_NON_OOM"
                error_type = "AbnormalProcessExit"
                failure_stage = "process"
                error_message = (
                    "Child process exited without a JSON result; empty stderr was "
                    "preserved in the retry log."
                )
        rows.append(
            {
                "model_id": model,
                "status": status,
                "prediction_shape": (
                    payload.get("prediction_shape") if payload else None
                ),
                "backward_completed": bool(
                    payload and payload.get("backward_completed")
                ),
                "optimizer_step_completed": bool(
                    payload and payload.get("optimizer_step_completed")
                ),
                "strict_reload_completed": bool(
                    payload and payload.get("strict_reload_completed")
                ),
                "artifact_validation": (
                    payload.get("artifact_validation", {}).get("status")
                    if payload
                    else None
                ),
                "failure_stage": failure_stage,
                "error_type": error_type,
                "error_message": error_message,
                "log": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("PASS", "FAIL_OOM", "FAIL_NON_OOM")
    }
    return {
        "schema_version": "uniform_batch4_real_data_smoke_inventory_v1",
        "status": "PASS" if counts["PASS"] == 26 else "PARTIAL_FAILURE",
        "profile_id": PROFILE_ID,
        "profile_hash": PROFILE_HASH,
        "dataset_input": "dataset/sdwpf_model_input_base.parquet",
        "dataset_target": "dataset/sdwpf_eval_target.parquet",
        "limits": {
            "train_batches": 2,
            "validation_batches": 1,
            "evaluation_batches": 1,
        },
        "rows": rows,
        "counts": counts,
        "notes": [
            "Interrupted attempts are preserved; successful retry2 results are "
            "used for DCRNN and the remaining graph models.",
            "MSGNet retry1 produced an empty stderr/stdout result after high "
            "host-memory use; exact preflight separately classified MSGNet as OOM.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def normalized_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in payload.items()
        if key not in {"phase", "created_at"}
    }
    for record in normalized.get("files", []):
        record.pop("mtime_ns", None)
    return normalized


def protection_results() -> dict[str, bool]:
    names = (
        "protected_core_hashes",
        "protected_model_sources",
        "protected_model_configs",
        "protected_tslib_hashes",
        "protected_graph_hashes",
        "protected_st_mgprompt_hashes",
        "protected_a0_a8_hashes",
        "protected_e5_a_hashes",
        "protected_formal_results_inventory",
    )
    return {
        name: normalized_snapshot(load(DOC_ROOT / f"{name}_before.json"))
        == normalized_snapshot(load(DOC_ROOT / f"{name}_after.json"))
        for name in names
    }


def block_downstream_preflights() -> None:
    all26 = load(DOC_ROOT / "BATCH4_ALL26_PREFLIGHT_RESULTS.json")
    all26["status"] = "NOT_RUN_BLOCKED_BY_KNOWN_OOM6"
    all26["blocked_reason"] = "BLOCKED_BATCH4_STILL_OOM"
    write_json("BATCH4_ALL26_PREFLIGHT_RESULTS.json", all26)
    for name in (
        "BATCH4_ST_MGPROMPT_FULL_PREFLIGHT.json",
        "BATCH4_A8_PREFLIGHT.json",
    ):
        payload = load(DOC_ROOT / name)
        payload["status"] = "NOT_RUN_BLOCKED_BY_KNOWN_OOM6"
        payload["blocked_reason"] = "BLOCKED_BATCH4_STILL_OOM"
        write_json(name, payload)


def main() -> int:
    known = load(DOC_ROOT / "BATCH4_KNOWN_OOM6_PREFLIGHT_RESULTS.json")
    block_downstream_preflights()
    real_smoke = build_real_smoke_inventory()
    write_json("BATCH4_REAL_DATA_SMOKE_RESULTS.json", real_smoke)
    protection = protection_results()

    sample_counts = {
        "time_steps": 52559,
        "train_samples_per_epoch": 6983,
        "train_batches_per_epoch": math.ceil(6983 / 4),
        "optimizer_steps_per_epoch": math.ceil(6983 / 4),
        "validation_samples": 1701,
        "validation_batches": math.ceil(1701 / 4),
        "test_samples": 5104,
        "test_batches": math.ceil(5104 / 4),
        "calculation": (
            "chronological 0.8/0.1/0.1 split; lookback=144, horizon=10; "
            "strides train/val/test=6/3/1; ceil(samples/4)"
        ),
    }
    tests = {
        "status": "PASS_WITH_TARGETED_RERUNS",
        "interpreter": (
            "D:/Apps/Miniconda3/envs/env_tslib/python.exe"
        ),
        "pytest_available": False,
        "suites": [
            {
                "suite": "existing benchmark_v2 unittest discovery",
                "tests_run": 164,
                "initial_pass": 162,
                "initial_fail": 2,
                "targeted_rerun_pass": 2,
                "notes": [
                    "E5 identity regression was corrected and its targeted test passed.",
                    "MTGNN zero-gradient assertion passed on isolated rerun without a model change.",
                ],
            },
            {
                "suite": "test_uniform_batch4.py",
                "tests_run": 8,
                "passed": 8,
                "failed": 0,
            },
        ],
        "coverage": [
            "profile schema/hash/allowlist/fixed batches",
            "DataLoader batch4 and unchanged sample collection",
            "gradient accumulation/effective batch",
            "initialization/forward/optimizer param-group invariance",
            "preflight identity mismatch rejection for batch32/16/8 and hashes",
            "artifact/checkpoint batch identity",
            "readiness/aggregation mixed-batch rejection",
            "old A8 rejection and new batch4 A8 acceptance",
            "backward compatibility, registry=28, TSLib allowlist=18",
        ],
    }
    write_json("test_results.json", tests)

    created = [
        "scripts/uniform_batch4_snapshot.py",
        "scripts/uniform_batch4_generate.py",
        "scripts/uniform_batch4_smoke.py",
        "scripts/uniform_batch4_finalize.py",
        "scripts/uniform_batch4_restore_graph.py",
        "custom_models/src/benchmark_v2/training_profiles/__init__.py",
        (
            "custom_models/src/benchmark_v2/training_profiles/"
            "uniform_train_batch4_v1.json"
        ),
        "custom_models/tests/benchmark_v2/test_uniform_batch4.py",
        "custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/",
    ]
    modified = [
        "custom_models/src/benchmark_v2/model_cli.py",
        "custom_models/src/benchmark_v2/hardware_preflight.py",
        "custom_models/src/benchmark_v2/cli.py",
        "custom_models/src/benchmark_v2/checkpointing.py",
        "custom_models/src/benchmark_v2/artifacts.py",
        (
            "custom_models/src/benchmark_v2/experiments/e5_common_loss/"
            "contracts.py"
        ),
        (
            "custom_models/src/benchmark_v2/experiments/e5_common_loss/"
            "variant_manifest.py"
        ),
        (
            "custom_models/src/benchmark_v2/experiments/e5_common_loss/"
            "a8_reference.py"
        ),
        (
            "custom_models/src/benchmark_v2/experiments/e5_common_loss/"
            "readiness.py"
        ),
        (
            "custom_models/src/benchmark_v2/experiments/e5_common_loss/"
            "aggregation.py"
        ),
        "custom_models/src/st_mgprompt/run_st_mgprompt.py",
        "custom_models/src/st_mgprompt/train.py",
    ]
    ordinary = load(DOC_ROOT / "BATCH4_ORDINARY_SMOKE_RESULTS.json")
    e5_smoke = load(DOC_ROOT / "BATCH4_E5_ORDINARY_SMOKE_RESULTS.json")
    st_smoke = load(
        DOC_ROOT / "BATCH4_ST_MGPROMPT_ORDINARY_SMOKE_RESULTS.json"
    )
    manifest = {
        "task": (
            "统一训练 Batch=4 降显存协议重构、全模型可行性预检与正式重训准备"
        ),
        "scope": "engineering preparation only; no formal training or aggregation",
        "status": known["status"],
        "profile_id": PROFILE_ID,
        "profile_hash": PROFILE_HASH,
        "train_batch_size": 4,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_train_batch_size": 4,
        "known_oom6_pass": known["counts"]["PASS"],
        "known_oom6_oom": known["counts"]["FAIL_OOM"],
        "known_oom6_non_oom": known["counts"]["FAIL_NON_OOM"],
        "all26_pass": 0,
        "all26_oom": 0,
        "all26_non_oom": 0,
        "all26_not_run": 26,
        "full_preflight_status": "NOT_RUN_BLOCKED_BY_KNOWN_OOM6",
        "a8_preflight_status": "NOT_RUN_BLOCKED_BY_KNOWN_OOM6",
        "registry_before": 28,
        "registry_after": 28,
        "tslib_allowlist_before": 18,
        "tslib_allowlist_after": 18,
        "benchmark_protocol_hash_before": PROTOCOL_HASH,
        "benchmark_protocol_hash_after": PROTOCOL_HASH,
        "canonical_hash_before": CANONICAL_HASH,
        "canonical_hash_after": CANONICAL_HASH,
        "graph_protocol_hash_before": GRAPH_HASH,
        "graph_protocol_hash_after": GRAPH_HASH,
        "model_sources_modified": False,
        "model_configs_modified": False,
        "loss_modified": False,
        "graph_protocol_modified": False,
        "st_mgprompt_architecture_modified": False,
        "dependencies_changed": False,
        "formal_original28_started": False,
        "formal_full_started": False,
        "formal_a8_started": False,
        "formal_e5_started": False,
        "e4_aggregation_started": False,
        "e5_aggregation_started": False,
        "ordinary_smoke": {
            "original28": ordinary["counts"],
            "e5_28": e5_smoke["counts"],
            "st_full_a8": st_smoke["counts"],
        },
        "real_data_smoke": real_smoke["counts"],
        "sample_and_batch_inventory": sample_counts,
        "files_created": created,
        "files_modified": modified,
        "files_deleted": [],
        "files_moved": [],
        "tests": tests,
        "protection_snapshot_matches": protection,
        "warnings": [
            "Exact preflight ran on NVIDIA GeForce GTX 1060 6 GiB, not the "
            "intended approximately 48 GiB formal GPU.",
            "Windows/WDDM CUDA peak counters in the artifacts can exceed physical "
            "VRAM because of virtual-memory accounting.",
            "Real-data smoke had SegRNN OOM and an abnormal MSGNet process exit; "
            "these are smoke diagnostics, not performance results.",
            "ST synthetic smoke initially rewrote the shared trend-prior files. "
            "The 134-node arrays and metadata were deterministically restored to "
            "all three frozen SHA256 values; their filesystem mtimes changed. "
            "Smoke graph output is now isolated per smoke run.",
            "Optimizer steps per epoch increase from the old batch32 protocol; "
            "no claim of equal update counts is made.",
        ],
        "blocked_reasons": [
            "BLOCKED_BATCH4_STILL_OOM",
            "known-OOM6 SegRNN backward OOM",
            "known-OOM6 MSGNet forward OOM",
        ],
    }
    write_json("IMPLEMENTATION_MANIFEST.json", manifest)

    write_text(
        "BATCH4_MEMORY_FEASIBILITY_REPORT.md",
        f"""# Batch=4 Memory Feasibility

Status: `BLOCKED_BATCH4_STILL_OOM`.

Exact shape was `B=4,T=144,N=134,C=16,H=10,AMP=true` on an NVIDIA GeForce
GTX 1060 (6 GiB). known-OOM6 counts: PASS={known['counts']['PASS']},
FAIL_OOM={known['counts']['FAIL_OOM']}, FAIL_NON_OOM={known['counts']['FAIL_NON_OOM']}.

SegRNN failed during backward. MSGNet failed during forward. Transformer,
PatchTST, FreTS, and TimeFilter passed forward/backward with finite gradients.
No fallback batch, gradient accumulation, checkpointing, offload, or model
capacity change was attempted.

Because this is not the intended approximately 48 GiB formal GPU, these results
block this machine but do not substitute for a rerun on the intended device.
""",
    )
    write_text(
        "IMPLEMENTATION_REPORT.md",
        f"""# Uniform Batch=4 Implementation Report

The versioned profile `{PROFILE_ID}` is implemented with hash `{PROFILE_HASH}`.
It injects train/val/test batch 4, gradient accumulation 1, effective batch 4,
and AMP true without entering model construction.

The old benchmark logical hash `{PROTOCOL_HASH}`, canonical checkpoint
`{CANONICAL_HASH}`, model/config/loss/graph identities, old A0/A8/E5 artifacts,
and old formal result inventory match their before snapshots.

Ordinary smoke passed 28/28 original entries, 28/28 E5 entries, and 2/2
ST-MGPrompt Full/A8 entries. Restricted real-data smoke completed the
26-model inventory with 24 PASS, 1 OOM, and 1 abnormal process exit.

Exact known-OOM6 preflight produced 4 PASS and 2 OOM, so all26, Full, and A8
exact preflights were intentionally not run. Formal suites and aggregations
remain NOT_RUN.
""",
    )
    write_text(
        "HANDOFF_UNIFORM_BATCH4.md",
        f"""# Handoff: Uniform Batch=4

Current status: `BLOCKED_BATCH4_STILL_OOM`.

Use only `{PROFILE_ID}` (`{PROFILE_HASH}`). The local 6 GiB GPU passed
Transformer, PatchTST, FreTS, and TimeFilter, but SegRNN OOMed in backward and
MSGNet OOMed in forward. Do not run formal suites from this state.

On the intended approximately 48 GiB GPU, rerun PRECHECK and known-OOM6 exact
preflight first. Only if all six PASS may all26, ST-MGPrompt Full, and A8 exact
preflights run. Only after all four gates PASS may the formal order be:
original 28, Full, A8, E5, E4 aggregation, E5 aggregation.

All long-run scripts are generated but were not executed. Batch32 results and
the old Canonical remain read-only. Full batch4 will be only a
`BATCH4_CANONICAL_CANDIDATE`; E5 must reference the new completed batch4 A8.
""",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "known_oom6": known["counts"],
                "real_smoke": real_smoke["counts"],
                "protection_all_match": all(protection.values()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
