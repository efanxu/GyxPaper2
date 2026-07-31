from __future__ import annotations

from typing import Any, Mapping


PERSISTENCE_CONFIG = {
    "model_id": "persistence",
    "baseline_rule": "repeat_last_visible_Patv_clean_for_input",
    "node_semantics": "node_shared",
    "cross_node_interaction": False,
    "optimizer": None,
    "scheduler": None,
    "epochs": None,
    "early_stopping": None,
}


def resolve_persistence_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **PERSISTENCE_CONFIG,
        "run_mode": run_mode,
        "horizon": int(protocol["max_pred_len"]),
        "lookback": int(protocol["lookback"]),
        "protocol_hash": protocol["protocol_hash"],
    }
