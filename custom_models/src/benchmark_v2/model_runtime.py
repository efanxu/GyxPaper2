from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import ModelUnavailableError
from .model_factories.registry import get_model_factory
from .registry import load_registry


@dataclass
class ModelRuntime:
    model_id: str
    model: Any
    adapter: Any
    effective_config: dict[str, Any]
    scaler_context: Any | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    graph_bundle: Any | None = None

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.model.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def inverse_target(self, values: Any) -> Any:
        if self.scaler_context is None:
            return values * self.target_std + self.target_mean
        return self.scaler_context.inverse_target(values)


def build_model_runtime(
    model_id: str,
    protocol: Mapping[str, Any],
    *,
    run_mode: str,
    input_scaler: Any | None = None,
    target_scaler: Any | None = None,
    ma_window: int | None = None,
) -> ModelRuntime:
    registry = load_registry()
    entry = registry.get(model_id)
    runtime_status = str(entry.runtime_status)
    allow_blocked_smoke = (
        run_mode == "smoke" and runtime_status.startswith("BLOCKED_")
    )
    if not runtime_status.startswith("AVAILABLE_") and not allow_blocked_smoke:
        raise ModelUnavailableError(
            f"{entry.display_name} is unavailable; planned_stage={entry.planned_stage}. "
            "No formal run was started."
        )
    factory = get_model_factory(entry.canonical_id)
    config = factory.resolve_config(
        protocol, run_mode=run_mode, ma_window=ma_window
    )
    graph_bundle = None
    graph_model_ids = {
        "gcn",
        "stgcn",
        "dcrnn",
        "graph_wavenet",
        "mtgnn",
        "agcrn",
        "stid",
    }
    if entry.canonical_id in graph_model_ids:
        graph_bundle = factory.load_graph_bundle()
        model = factory.create_model(
            config=config,
            protocol=protocol,
            graph_bundle=graph_bundle,
            _allow_blocked_smoke=allow_blocked_smoke,
        )
    else:
        model = factory.create_model(
            config=config,
            protocol=protocol,
            _allow_blocked_smoke=allow_blocked_smoke,
        )
    if entry.canonical_id in graph_model_ids or entry.canonical_id not in {
        "persistence",
        "moving_average",
    }:
        target_mean, target_std = 0.0, 1.0
        if target_scaler is not None:
            target_mean = float(np.asarray(target_scaler.mean).reshape(-1)[0])
            target_std = float(np.asarray(target_scaler.std).reshape(-1)[0])
        if entry.canonical_id in {"gcn", "stgcn", "dcrnn"}:
            assert graph_bundle is not None
            adapter = factory.create_adapter(
                model_id=entry.canonical_id,
                graph_bundle=graph_bundle,
                protocol=protocol,
                config=config,
            )
            factory.apply_graph_identity(config, graph_bundle)
            config["source_hash"] = entry.values.get("source_sha256")
            config["source_closure_manifest"] = entry.values.get(
                "source_closure_manifest"
            )
        elif entry.canonical_id in {"graph_wavenet", "mtgnn", "agcrn", "stid"}:
            assert graph_bundle is not None
            adapter = factory.create_adapter(
                model_id=entry.canonical_id,
                graph_bundle=graph_bundle,
                protocol=protocol,
                config=config,
            )
            factory.apply_graph_identity(
                config, graph_bundle, entry.canonical_id
            )
            config["source_hash"] = entry.values.get("source_sha256")
            config["source_closure_hash"] = entry.values.get("source_sha256")
            config["source_closure_manifest"] = entry.values.get(
                "source_closure_manifest"
            )
        else:
            adapter = factory.create_adapter(
                protocol=protocol,
                config=config,
            )
            provenance = dict(
                getattr(model, "_benchmark_v2_upstream_provenance", {})
            )
            config.update(
                {
                    "upstream_project": provenance.get("upstream_project"),
                    "upstream_source_path": provenance.get("source_path"),
                    "upstream_source_relative_path": provenance.get(
                        "source_relative_path"
                    ),
                    "upstream_source_sha256": provenance.get("source_sha256"),
                    "upstream_license_path": provenance.get("license_path"),
                    "upstream_license_sha256": provenance.get("license_sha256"),
                    "source_modified": provenance.get("source_modified", False),
                }
            )
        parameter_count = sum(int(value.numel()) for value in model.parameters())
        trainable_parameter_count = sum(
            int(value.numel())
            for value in model.parameters()
            if value.requires_grad
        )
        config["parameter_count"] = parameter_count
        config["trainable_parameter_count"] = trainable_parameter_count
        return ModelRuntime(
            model_id=entry.canonical_id,
            model=model,
            adapter=adapter,
            effective_config=config,
            target_mean=target_mean,
            target_std=target_std,
            graph_bundle=graph_bundle,
        )
    if input_scaler is None or target_scaler is None:
        raise ValueError(
            f"{entry.display_name} requires explicit train-only input and target scalers."
        )
    adapter = factory.create_adapter(
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        protocol=protocol,
        config=config,
    )
    return ModelRuntime(
        model_id=entry.canonical_id,
        model=model,
        adapter=adapter,
        effective_config=config,
        scaler_context=getattr(adapter, "scaler_context", None),
    )
