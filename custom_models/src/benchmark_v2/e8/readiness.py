from __future__ import annotations

from typing import Any


def build_readiness(
    evidence_manifest: dict[str, Any], *, representation_status: str, cross_status: str,
    macro_status: str, st_status: str, paired_status: str, grouped_status: str,
) -> dict[str, Any]:
    evidence = evidence_manifest["evidence"]
    internal = [row for row in evidence if row["evidence_id"].startswith("E8_INTERNAL_")]
    external = [row for row in evidence if row["evidence_id"].startswith("E8_EXTERNAL_")]
    internal_ready = sum(bool(row["ready"]) for row in internal)
    external_ready = sum(bool(row["ready"]) for row in external)
    core_ready = internal_ready + external_ready
    return {
        "schema_version": "e8_readiness_v1",
        "INTERNAL_PROMPT_FUSION_READY": f"{internal_ready}/5",
        "ORIGINAL26_EXTERNAL_READY": f"{external_ready}/6",
        "CORE_E8_READY": f"{core_ready}/11",
        "core_ready": core_ready == 11,
        "REPRESENTATION_DIAGNOSTICS_READY": representation_status,
        "CROSS_FUSION_DIAGNOSTICS_READY": cross_status,
        "MACRO_PROMPT_DIAGNOSTICS_READY": macro_status,
        "ST_PROMPT_DIAGNOSTICS_READY": st_status,
        "PAIRED_WINDOW_ANALYSIS_READY": paired_status,
        "GROUPED_ANALYSIS_READY": grouped_status,
        "blockers": [{"evidence_id": row["evidence_id"], "errors": row["validation_errors"]} for row in evidence if not row["ready"]],
        "fail_closed": core_ready != 11,
    }
