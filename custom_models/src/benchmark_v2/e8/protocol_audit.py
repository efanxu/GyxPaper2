from __future__ import annotations

from .constants import BASE_PROTOCOL, BATCH4_PROFILE, EXPECTED_BATCH, EXPECTED_PROTOCOL
from .io_utils import read_json


def audit_protocol() -> dict:
    protocol = read_json(BASE_PROTOCOL)
    profile = read_json(BATCH4_PROFILE)
    checks = {}
    for key, expected in EXPECTED_PROTOCOL.items():
        actual = protocol.get(key)
        checks[f"protocol.{key}"] = {"expected": expected, "actual": actual, "passed": actual == expected}
    for key, expected in EXPECTED_BATCH.items():
        actual = profile.get(key)
        checks[f"batch.{key}"] = {"expected": expected, "actual": actual, "passed": actual == expected}
    return {
        "schema_version": "e8_protocol_audit_v1",
        "status": "PASS" if all(row["passed"] for row in checks.values()) else "FAIL",
        "protocol_id": protocol.get("protocol_id"),
        "batch_profile_id": profile.get("profile_id"),
        "ordered_input_features": protocol.get("ordered_input_features"),
        "checks": checks,
        "old_batch32_accepted": False,
    }
