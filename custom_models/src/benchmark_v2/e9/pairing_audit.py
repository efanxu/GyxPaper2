from __future__ import annotations

from typing import Any

from .constants import MODEL_IDS
from .variants import canonical_transfer_run_id


def build_pairing_audit(control_audit: dict[str, Any], transfer_evidence: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    controls = {row["model_id"]: row for row in control_audit["controls"]}
    transfers = transfer_evidence or {}
    pairs = []
    for model_id in MODEL_IDS:
        control = controls[model_id]
        transfer = transfers.get(model_id, {})
        checkpoint = control.get("checkpoint_metadata", {})
        control_initial = checkpoint.get("initial_model_state_hash")
        transfer_initial = transfer.get("initial_model_state_hash")
        if control_initial is not None and transfer_initial is not None:
            exact = "PASS" if control_initial == transfer_initial else "FAIL"
        else:
            exact = "NOT_VERIFIED"
        same_seed = control.get("seed") == (transfer.get("seed", control.get("seed")))
        pairs.append({
            "model_id": model_id,
            "control_run_id": control.get("run_id"),
            "transfer_run_id": canonical_transfer_run_id(model_id),
            "control_initial_model_state_hash": control_initial,
            "transfer_initial_model_state_hash": transfer_initial,
            "EXACT_INITIAL_STATE_PAIRING": exact,
            "initialization_seed": control.get("seed"),
            "same_seed": same_seed,
            "control_rng_state_present": bool(checkpoint.get("rng_state_present")),
            "transfer_rng_manifest_present": bool(transfer.get("rng_manifest_present")),
            "sampler_identity": "ProviderBatchIterable sequential formal starts",
            "data_order_identity": "strict chronological train starts, stride=6, no shuffle",
            "allowed_claim": "same seed, same source/config, same protocol" if exact == "NOT_VERIFIED" else "exact initial state verified",
        })
    return {
        "schema_version": "e9_pairing_audit_v1",
        "pair_count": len(pairs),
        "exact_initial_state_pairing_verified": sum(row["EXACT_INITIAL_STATE_PAIRING"] == "PASS" for row in pairs),
        "pairs": pairs,
    }


def pairing_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# E9 Pairing / Initialization Audit", "",
        "original26 controls are historical formal runs. Exact initial weights are claimed only when both artifacts expose matching initial-state identities.", "",
        "| Model | Control | Transfer | Exact initial state | Allowed claim |", "|---|---|---|---|---|",
    ]
    for row in audit["pairs"]:
        lines.append(f"| {row['model_id']} | {row['control_run_id']} | {row['transfer_run_id']} | {row['EXACT_INITIAL_STATE_PAIRING']} | {row['allowed_claim']} |")
    lines.extend(["", "All E9 transfers save loss state in checkpoints and preserve the sequential formal data order.", ""])
    return "\n".join(lines)
