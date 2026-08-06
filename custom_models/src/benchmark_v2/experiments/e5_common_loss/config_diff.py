from __future__ import annotations

from typing import Any, Mapping


ALLOWED_DIFF_PATHS = {
    "experiment_profile_id",
    "loss.id",
    "loss.source_function",
    "run_id",
    "output_root",
    "preflight_metadata",
    "provenance",
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    return {prefix: value}


def _allowed(path: str) -> bool:
    return any(path == item or path.startswith(item + ".") for item in ALLOWED_DIFF_PATHS)


def compare_configs(base: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    def normalize(config: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(config)
        if "loss" in result and not isinstance(result["loss"], Mapping):
            result["loss"] = {"id": result["loss"]}
        return result

    left, right = _flatten(normalize(base)), _flatten(normalize(variant))
    differences = {
        key: {"base": left.get(key), "e5": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    unexpected = sorted(key for key in differences if not _allowed(key))
    return {
        "status": "PASS" if not unexpected else "BLOCKED_CONFIG_DIFF",
        "differences": differences,
        "unexpected_differences": unexpected,
        "allowed_paths": sorted(ALLOWED_DIFF_PATHS),
    }


def state_dict_equal(left, right) -> bool:
    import torch
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[key].detach().cpu(), right_state[key].detach().cpu())
        for key in left_state
    )


def optimizer_group_signature(optimizer) -> list[dict[str, Any]]:
    result = []
    for group in optimizer.param_groups:
        result.append(
            {
                key: value
                for key, value in group.items()
                if key != "params"
            }
            | {"parameter_count": sum(int(item.numel()) for item in group["params"])}
        )
    return result
