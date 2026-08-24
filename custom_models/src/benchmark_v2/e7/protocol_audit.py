from __future__ import annotations

from .constants import (
    BASE_PROTOCOL,
    BATCH4_PROFILE,
    EXPECTED_BATCH,
    EXPECTED_GRAPH,
    EXPECTED_PROTOCOL,
    GRAPH_PROTOCOL,
    NODE_ORDER,
)
from .io_utils import read_json


def audit_protocol() -> dict:
    protocol = read_json(BASE_PROTOCOL)
    profile = read_json(BATCH4_PROFILE)
    graph = read_json(GRAPH_PROTOCOL)
    node_order = read_json(NODE_ORDER)
    checks = {}
    for key, expected in EXPECTED_PROTOCOL.items():
        checks[f"protocol.{key}"] = {"expected": expected, "actual": protocol.get(key), "passed": protocol.get(key) == expected}
    for key, expected in EXPECTED_BATCH.items():
        checks[f"batch.{key}"] = {"expected": expected, "actual": profile.get(key), "passed": profile.get(key) == expected}
    for key, expected in EXPECTED_GRAPH.items():
        checks[f"graph.{key}"] = {"expected": expected, "actual": graph.get(key), "passed": graph.get(key) == expected}
    ordered = node_order.get("ordered_node_ids", node_order.get("node_ids", []))
    checks["node_order.TurbID_1_134"] = {
        "expected": list(range(1, 135)),
        "actual": ordered,
        "passed": ordered == list(range(1, 135)),
    }
    return {
        "schema_version": "e7_protocol_audit_v1",
        "status": "PASS" if all(item["passed"] for item in checks.values()) else "FAIL",
        "protocol_id": protocol.get("protocol_id"),
        "batch_profile_id": profile.get("profile_id"),
        "graph_id": graph.get("graph_id"),
        "ordered_input_features": protocol.get("ordered_input_features"),
        "checks": checks,
    }
