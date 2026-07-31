from __future__ import annotations

import json
from pathlib import Path


DOC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def read_json(name: str):
    return json.loads((DOC_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    graph = read_json("graph_hash_manifest.json")
    diagnostics = read_json("graph_build_diagnostics.json")
    checks = read_json("graph_protocol_check_result.json")
    tests = read_json("test_results.json")
    summary = {
        "task": "E3-A",
        "status": (
            "PASS"
            if checks.get("status") == "PASS"
            and tests.get("status") == "PASS"
            else "FAIL"
        ),
        "graph_id": checks["graph_id"],
        "node_count": checks["node_count"],
        "selected_k": checks["selected_k"],
        "node_order_hash": graph["node_order_hash"],
        "graph_bundle_hash": graph["graph_bundle_hash"],
        "graph_protocol_hash": graph["graph_protocol_hash"],
        "connected_component_count": diagnostics[
            "connected_component_count"
        ],
        "tests": tests,
    }
    (DOC_DIR / "e3_a_collected_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    protection = {}
    for group in (
        "core",
        "tslib",
        "benchmark_frozen",
        "e1_a",
        "e1_b",
        "e2_a",
        "e2_b",
        "e2_c",
        "e2_d",
    ):
        protection[group] = read_json(
            f"protected_{group}_hashes_after.json"
        )
    created_roots = [
        DOC_DIR,
        PROJECT_ROOT / "custom_models/src/benchmark_v2/graph",
        PROJECT_ROOT / "custom_models/src/benchmark_v2/protocol/graph_v1",
        PROJECT_ROOT
        / "custom_models/results_smoke/benchmark_v2/e3_a_contract",
    ]
    created = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for root in created_roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    created.update(
        {
            "custom_models/src/benchmark_v2/protocol/graph_protocol_v1.json",
            "custom_models/tests/benchmark_v2/test_e3_a_graph_protocol.py",
            "custom_models/docs/benchmark_v2/E3_A/E3_A_IMPLEMENTATION_MANIFEST.json",
            "custom_models/docs/benchmark_v2/E3_A/e3_a_collected_summary.json",
        }
    )
    before = protection["core"]
    benchmark_after = protection["benchmark_frozen"]
    registry_counts = {
        "total_entries": 28,
        "formal_runnable_trainable": 19,
        "locally_full_shape_verified": 9,
        "hardware_preflight_required": 10,
        "non_trainable_available": 2,
        "true_blocked_or_unavailable": 7,
    }
    manifest = {
        "task": "E3-A",
        "scope": (
            "graph protocol, canonical node identity, GraphSpec/GraphBundle "
            "contract, and STCN/STGCN identity only"
        ),
        "task_start_existing_changes": (
            "UNKNOWN: project root has no Git metadata"
        ),
        "prerequisite_files_found": [
            "custom_models/docs/benchmark_v2/E2_A/HANDOFF_E2_A.md",
            "custom_models/docs/benchmark_v2/E2_A/E2_A_HARDWARE_PREFLIGHT_POLICY.md",
            "custom_models/docs/benchmark_v2/E2_B/HANDOFF_E2_B.md",
            "custom_models/docs/benchmark_v2/E2_B/E2_B_IMPLEMENTATION_REPORT.md",
            "custom_models/docs/benchmark_v2/E2_B/E2_B_RUNBOOK.md",
            "custom_models/docs/benchmark_v2/E2_C/HANDOFF_E2_C.md",
            "custom_models/docs/benchmark_v2/E2_C/E2_C_IMPLEMENTATION_REPORT.md",
            "custom_models/docs/benchmark_v2/E2_C/E2_C_RUNBOOK.md",
            "custom_models/docs/benchmark_v2/E2_D/HANDOFF_E2_D.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_IMPLEMENTATION_REPORT.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_CONFIG_RESOLUTION.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_MODEL_SPECIFICATIONS.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_TSLIB_SOURCE_AUDIT.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_BATCH_ISOLATION_AUDIT.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_CORRELATION_FAIRNESS_AUDIT.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_RUNBOOK.md",
            "custom_models/docs/benchmark_v2/E2_D/E2_D_IMPLEMENTATION_MANIFEST.json",
            "custom_models/docs/benchmark_v2/E2_D/test_results.json",
        ],
        "prerequisite_files_missing": ["实验总Plan.md", "HANDOFF.md"],
        "graph_protocol_status": "FROZEN",
        "graph_id": checks["graph_id"],
        "selected_k": checks["selected_k"],
        "node_count": checks["node_count"],
        "node_order_hash": graph["node_order_hash"],
        "location_source_hash": graph["location_source_sha256"],
        "graph_bundle_hash": graph["graph_bundle_hash"],
        "graph_protocol_hash": graph["graph_protocol_hash"],
        "stcn_stgcn_final_canonical_id": "stgcn",
        "stcn_stgcn_identity_status": (
            "RESOLVED_CANONICAL_TARGET_STGCN_SOURCE_MISSING"
        ),
        "stcn_stgcn_implementation_status": "MISSING_IMPLEMENTATION",
        "registry_counts_before": registry_counts,
        "registry_counts_after": registry_counts,
        "allowlist_before": 18,
        "allowlist_after": 18,
        "formal_training_started": False,
        "formal_evaluation_started": False,
        "target_high_memory_preflight_started": False,
        "graph_model_implementation_started": False,
        "e3_b_started": False,
        "formal_output_root_created": False,
        "protocol_hash_before": (
            "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b"
        ),
        "protocol_hash_after": (
            "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b"
        ),
        "protocol_file_sha256_before": before["protocol_file_sha256"],
        "protocol_file_sha256_after": protection["core"][
            "protocol_file_sha256"
        ],
        "canonical_sha256_before": before[
            "canonical_checkpoint_sha256"
        ],
        "canonical_sha256_after": protection["core"][
            "canonical_checkpoint_sha256"
        ],
        "tslib_modified": bool(protection["tslib"]["mismatches"]),
        "st_mgprompt_modified": bool(protection["core"]["mismatches"]),
        "prior_models_modified": any(
            protection[name]["mismatches"]
            for name in (
                "e1_a",
                "e1_b",
                "e2_a",
                "e2_b",
                "e2_c",
                "e2_d",
            )
        ),
        "dependencies_changed": False,
        "files_created_count": len(created),
        "files_created": sorted(created),
        "files_modified_count": 2,
        "files_modified": [
            "custom_models/src/benchmark_v2/cli.py",
            "custom_models/src/benchmark_v2/registry/benchmark_registry.json",
        ],
        "files_deleted": [],
        "files_moved": [],
        "commands_executed": [
            "UTF-8 read of attached E3-A prompt and all required available prerequisites",
            "nine-group E3-A before protection snapshot",
            "read-only whole-workspace graph and STCN/STGCN audit",
            "parquet key/schema and location metadata audit",
            "CPU float64 graph_protocol_v1 initial freeze",
            "two consecutive in-memory graph rebuilds",
            "python -m compileall -q custom_models/src/benchmark_v2",
            "python -m unittest custom_models.tests.benchmark_v2.test_e3_a_graph_protocol -v",
            "python -m unittest discover -s custom_models/tests/benchmark_v2 -p test_*.py",
            "run_benchmark.py protocol-check",
            "run_benchmark.py graph-protocol-check",
            "CPU-only dummy graph contract smoke",
            "nine-group E3-A after protection snapshot",
        ],
        "tests": tests,
        "protection": {
            name: {
                "mismatches": payload["mismatches"],
                "protocol_matches_before": payload[
                    "protocol_matches_before"
                ],
                "canonical_matches_before": payload[
                    "canonical_matches_before"
                ],
                "formal_output_root_matches_before": payload[
                    "formal_output_root_matches_before"
                ],
            }
            for name, payload in protection.items()
        },
        "expected_in_scope_protected_mismatches": benchmark_after[
            "mismatches"
        ],
        "warnings": [
            "实验总Plan.md and HANDOFF.md are NOT_FOUND and were not reconstructed.",
            "The project has no Git metadata; existing task-start changes are UNKNOWN.",
            "Local evidence fixes Cartesian x/y semantics but does not declare coordinate/elevation units; units are SOURCE_UNIT_UNSPECIFIED.",
            "benchmark_frozen snapshot changes are limited to CLI graph-protocol-check dispatch and the expressly permitted STCN/STGCN Registry metadata resolution."
        ],
        "blocked_reasons": [
            "STGCN runtime remains BLOCKED_UNIMPLEMENTED because no verifiable local implementation exists."
        ],
        "not_run": {
            "GCN implementation": "NOT_STARTED",
            "STGCN implementation": "NOT_STARTED",
            "DCRNN implementation": "NOT_STARTED",
            "Graph WaveNet implementation": "NOT_STARTED",
            "MTGNN implementation": "NOT_STARTED",
            "AGCRN implementation": "NOT_STARTED",
            "STID implementation": "NOT_STARTED",
            "formal training": "NOT_RUN",
            "formal evaluation": "NOT_RUN",
            "target-machine preflight": "NOT_RUN",
            "E3-B": "NOT_STARTED",
        },
    }
    (DOC_DIR / "E3_A_IMPLEMENTATION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
