from __future__ import annotations

from typing import Any

from .constants import MODEL_IDS


def build_readiness(portability: dict[str, Any], evidence_manifest: dict[str, Any]) -> dict[str, Any]:
    evidence = evidence_manifest["evidence"]
    controls = [row for row in evidence if row["evidence_role"] == "ORIGINAL26_CONTROL"]
    transfers = [row for row in evidence if row["evidence_role"] == "E9_TRANSFER"]
    control_ready = sum(bool(row.get("ready")) for row in controls)
    transfer_ready = sum(bool(row.get("ready")) for row in transfers)
    pair_rows = []
    for model_id in MODEL_IDS:
        control = next(row for row in controls if row["model_id"] == model_id)
        transfer = next(row for row in transfers if row["model_id"] == model_id)
        pair_ready = bool(control.get("ready") and transfer.get("ready") and transfer.get("loss_only_diff", {}).get("loss_only_diff_valid"))
        pair_rows.append({
            "model_id": model_id, "control_ready": bool(control.get("ready")),
            "transfer_ready": bool(transfer.get("ready")),
            "loss_only_diff_valid": bool(transfer.get("loss_only_diff", {}).get("loss_only_diff_valid")),
            "pair_status": "READY" if pair_ready else "NOT_READY", "ready": pair_ready,
        })
    pairing_ready = sum(row["ready"] for row in pair_rows)
    core_ready_count = control_ready + transfer_ready
    return {
        "schema_version": "e9_readiness_v1",
        "IMPLEMENTATION_READY": True,
        "PORTABILITY_AUDIT_READY": "PASS" if portability.get("e9_b_allowed") else "FAIL",
        "ORIGINAL_MASKED_MSE_REFERENCE_READY": f"{control_ready}/6",
        "MSMG_DWU_TRANSFER_READY": f"{transfer_ready}/6",
        "LOSS_ONLY_PAIRING_READY": f"{pairing_ready}/6",
        "CORE_E9_READY": f"{core_ready_count}/12",
        "CORE_E9_READY_BOOL": core_ready_count == 12 and pairing_ready == 6,
        "FULL_MS_MG_DWU_TRANSFER_READY": bool(portability.get("FULL_MS_MG_DWU_TRANSFER_READY")),
        "PORTABLE_SUBSET_TRANSFER_READY": bool(portability.get("PORTABLE_SUBSET_TRANSFER_READY")),
        "STMG_REFERENCE_READY": "0/2",
        "pairs": pair_rows,
        "e5_consumed": False,
    }
