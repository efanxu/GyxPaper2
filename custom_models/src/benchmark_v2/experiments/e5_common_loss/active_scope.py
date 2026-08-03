from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ...runtime import PROJECT_ROOT


ACTIVE_SCOPE_POINTER_PATH = (
    PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_ACTIVE_SCOPE.json"
)
ACTIVE_SCOPE_ID = "e5_batch4_scope27_seed2026"


class ActiveScopeError(ValueError):
    """Raised when the current E5 active-scope pointer is unsafe or invalid."""


def _resolve_repo_relative(value: str, *, field: str) -> Path:
    raw = str(value).replace("\\", "/")
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raise ActiveScopeError(f"Active scope {field} must be repository-relative.")
    root = PROJECT_ROOT.resolve()
    candidate = (root / Path(raw)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ActiveScopeError(
            f"Active scope {field} escapes the repository: {value}"
        ) from exc
    return candidate


def active_scope_path(pointer: Mapping[str, Any], field: str) -> Path:
    value = pointer.get(field)
    if not isinstance(value, str) or not value:
        raise ActiveScopeError(f"Active scope pointer is missing {field!r}.")
    path = _resolve_repo_relative(value, field=field)
    if not path.is_file():
        raise ActiveScopeError(
            f"Active scope pointer {field} does not exist: {path}"
        )
    return path


def load_active_scope_pointer(
    path: str | Path = ACTIVE_SCOPE_POINTER_PATH,
) -> dict[str, Any]:
    pointer_path = Path(path).resolve()
    if pointer_path != ACTIVE_SCOPE_POINTER_PATH.resolve():
        raise ActiveScopeError(
            f"Only the repository active scope pointer is valid: {ACTIVE_SCOPE_POINTER_PATH}"
        )
    if not pointer_path.is_file():
        raise ActiveScopeError(f"Active scope pointer is missing: {pointer_path}")
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ActiveScopeError(
            f"Active scope pointer cannot be parsed: {pointer_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ActiveScopeError("Active scope pointer root must be an object.")
    if payload.get("status") != "ACTIVE" or payload.get("scope_id") != ACTIVE_SCOPE_ID:
        raise ActiveScopeError("Current active E5 scope pointer is not scope27 ACTIVE.")
    for field in (
        "manifest",
        "run_id_map",
        "readiness_policy",
        "gate_source",
        "linux_launcher",
        "linux_autoshutdown_launcher",
        "runbook",
        "handoff",
    ):
        active_scope_path(payload, field)
    return payload
