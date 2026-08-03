from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ModelUnavailableError
from ..model_factories.registry import get_model_factory


REGISTRY_PATH = Path(__file__).with_name("benchmark_registry.json")
REQUIRED_FIELDS = {
    "canonical_id", "display_name", "aliases", "category", "planned_stage", "audit_status", "runtime_status",
    "source_type", "source_path", "upstream_project", "license_path", "adapter_kind", "node_semantics",
    "requires_graph", "requires_exogenous_split", "requires_time_marks", "factory", "config_schema",
    "implementation_module", "expected_input_contract", "expected_output_contract", "known_issues",
    "known_smoke_status", "known_smoke_error", "supports_train", "supports_evaluate", "supports_non_trainable",
    "protocol_version", "notes",
}


@dataclass(frozen=True)
class RegistryEntry:
    values: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


class BenchmarkRegistry:
    """Explicit JSON registry. It never scans models or imports a framework."""

    def __init__(self, entries: list[RegistryEntry], source: Path = REGISTRY_PATH):
        self.entries = tuple(entries)
        self.source = source
        self._by_id = {entry.canonical_id: entry for entry in entries}
        self._by_alias = {}
        for entry in entries:
            for alias in entry.aliases:
                key = str(alias).casefold()
                if key in self._by_alias:
                    raise ValueError(f"Alias collision: {alias}")
                self._by_alias[key] = entry
        if len(self._by_id) != len(entries):
            raise ValueError("Canonical IDs must be unique.")
        names = [entry.display_name.casefold() for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError("Display names must be unique.")
        if len(entries) != 28:
            raise ValueError(f"Formal benchmark registry must contain 28 entries, got {len(entries)}")

    def list(self) -> list[RegistryEntry]:
        return list(self.entries)

    def statistics(self) -> dict[str, int]:
        statuses = [str(entry.runtime_status) for entry in self.entries]
        locally_verified = sum(
            status == "AVAILABLE_TRAINABLE" for status in statuses
        )
        preflight_required = sum(
            status == "AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED"
            for status in statuses
        )
        non_trainable = sum(
            status == "AVAILABLE_NON_TRAINABLE" for status in statuses
        )
        return {
            "total_entries": len(statuses),
            "available_locally_verified": locally_verified,
            "available_with_hardware_preflight": preflight_required,
            "non_trainable_available": non_trainable,
            "true_blocked_or_unavailable": (
                len(statuses) - locally_verified - preflight_required - non_trainable
            ),
        }

    def get(self, model_id: str) -> RegistryEntry:
        key = str(model_id)
        entry = self._by_id.get(key) or self._by_alias.get(key.casefold())
        if entry is None:
            raise KeyError(f"Unknown benchmark model: {model_id}")
        return entry

    def create_model(self, model_id: str, *args: Any, **kwargs: Any) -> Any:
        entry = self.get(model_id)
        allow_blocked = bool(kwargs.pop("_allow_blocked_smoke", False))
        status = str(entry.runtime_status)
        runnable = status.startswith("AVAILABLE_") or (
            allow_blocked and status.startswith("BLOCKED_")
        )
        if not entry.factory or not runnable:
            raise ModelUnavailableError(
                f"ModelUnavailableError: model={entry.canonical_id}\n"
                f"runtime_status={entry.runtime_status}\n"
                f"planned_stage={entry.planned_stage}\n"
                f"source was audited but no benchmark_v2 adapter is registered"
            )
        factory = get_model_factory(entry.canonical_id)
        create_model = getattr(factory, "create_model", None)
        if not callable(create_model):
            raise ModelUnavailableError(
                f"Audited model factory has no create_model function: {entry.canonical_id}"
            )
        return create_model(*args, **kwargs)


def load_registry(path: str | Path | None = None) -> BenchmarkRegistry:
    source = Path(path) if path else REGISTRY_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("registry_version") != "benchmark_registry_v1":
        raise ValueError("Unsupported benchmark registry version.")
    raw_entries = payload.get("models")
    if not isinstance(raw_entries, list):
        raise ValueError("Registry models must be a list.")
    for raw in raw_entries:
        missing = REQUIRED_FIELDS - set(raw)
        if missing:
            raise ValueError(f"Registry entry {raw.get('canonical_id')} missing {sorted(missing)}")
    return BenchmarkRegistry([RegistryEntry(dict(raw)) for raw in raw_entries], source=source)
