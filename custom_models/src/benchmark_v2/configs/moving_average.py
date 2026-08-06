from __future__ import annotations

from typing import Any, Mapping

from ..errors import ContractError


MOVING_AVERAGE_CONFIG = {
    "model_id": "moving_average",
    "baseline_rule": "repeat_physical_trailing_mean_Patv_clean_for_input",
    "ma_window": 144,
    "node_semantics": "node_shared",
    "cross_node_interaction": False,
    "optimizer": None,
    "scheduler": None,
    "epochs": None,
    "early_stopping": None,
}


def resolve_moving_average_config(
    protocol: Mapping[str, Any],
    *,
    run_mode: str = "formal",
    ma_window: int | None = None,
) -> dict[str, Any]:
    value = int(MOVING_AVERAGE_CONFIG["ma_window"] if ma_window is None else ma_window)
    formal_window = int(protocol["lookback"])
    if run_mode == "formal" and value != formal_window:
        raise ContractError(
            f"Formal MovingAverage ma_window is frozen at {formal_window}; got {value}."
        )
    if run_mode not in {"formal", "smoke"}:
        raise ContractError(f"Unknown MovingAverage run_mode: {run_mode}")
    return {
        **MOVING_AVERAGE_CONFIG,
        "run_mode": run_mode,
        "ma_window": value,
        "horizon": int(protocol["max_pred_len"]),
        "lookback": formal_window,
        "protocol_id": protocol["protocol_id"],
    }
