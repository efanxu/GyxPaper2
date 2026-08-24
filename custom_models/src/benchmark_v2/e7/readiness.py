from __future__ import annotations

from typing import Any


def build_readiness(evidence_manifest: dict[str, Any], *, grouped_status: str, diagnostics_status: str, node_status: str) -> dict[str, Any]:
    evidence = evidence_manifest["evidence"]
    internal = [row for row in evidence if row["evidence_id"].startswith("E7_INTERNAL_")]
    external = [row for row in evidence if row["evidence_id"].startswith("E7_EXTERNAL_")]
    internal_ready = sum(bool(row["ready"]) for row in internal)
    external_ready = sum(bool(row["ready"]) for row in external)
    core_ready = internal_ready + external_ready
    blockers = [
        {"evidence_id": row["evidence_id"], "errors": row["validation_errors"]}
        for row in evidence
        if not row["ready"]
    ]
    return {
        "schema_version": "e7_readiness_v1",
        "INTERNAL_GRAPH_READY": f"{internal_ready}/4",
        "ORIGINAL26_EXTERNAL_READY": f"{external_ready}/6",
        "CORE_E7_READY": f"{core_ready}/10",
        "core_ready": core_ready == 10,
        "GRAPH_DIAGNOSTICS_READY": diagnostics_status,
        "NODE_ANALYSIS_READY": node_status,
        "GROUPED_ANALYSIS_READY": grouped_status,
        "blockers": blockers,
        "fail_closed": core_ready != 10,
    }
