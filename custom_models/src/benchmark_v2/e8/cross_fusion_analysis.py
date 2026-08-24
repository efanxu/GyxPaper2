from __future__ import annotations

from typing import Any

import numpy as np


NOT_APPLICABLE = "NOT_APPLICABLE"


def implementation_audit(module: Any) -> dict[str, Any]:
    names = {name for name, _ in module.named_modules()}
    explicit_attention = any(name.endswith("macro_to_fine") or name.endswith("fine_to_coarse") for name in names)
    explicit_gate = any(name.endswith("gate") for name in names)
    return {
        "implementation_type": ["attention", "gate", "residual", "MLP", "additive"] if explicit_attention and explicit_gate else ["other"],
        "has_explicit_attention": explicit_attention, "has_explicit_gate": explicit_gate,
        "attention_fields_status": "EXTRACTABLE" if explicit_attention else NOT_APPLICABLE,
        "gate_fields_status": "EXTRACTABLE" if explicit_gate else NOT_APPLICABLE,
    }


def attention_summary(weights: np.ndarray | None, top_k: int = 5) -> dict[str, Any]:
    if weights is None:
        return {"status": NOT_APPLICABLE, "attention_entropy": NOT_APPLICABLE, "max_weight": NOT_APPLICABLE, "top_k_interaction": NOT_APPLICABLE}
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("Attention weights need at least query/key dimensions.")
    probs = np.clip(array, 1e-12, None)
    probs /= probs.sum(axis=-1, keepdims=True)
    return {
        "status": "OK", "attention_entropy": float((-(probs * np.log(probs)).sum(axis=-1)).mean()),
        "max_weight": float(probs.max(axis=-1).mean()),
        "top_k_interaction": float(np.sort(probs, axis=-1)[..., -min(top_k, probs.shape[-1]):].sum(axis=-1).mean()),
    }


def directional_statistics(coarse_to_fine: np.ndarray | None, fine_to_coarse: np.ndarray | None) -> dict[str, Any]:
    first, second = attention_summary(coarse_to_fine), attention_summary(fine_to_coarse)
    if first["status"] != "OK" or second["status"] != "OK":
        asymmetry = NOT_APPLICABLE
    else:
        a, b = first["max_weight"], second["max_weight"]
        asymmetry = (a - b) / max(a + b, 1e-12)
    return {"coarse_to_fine": first, "fine_to_coarse": second, "directional_asymmetry": asymmetry}
