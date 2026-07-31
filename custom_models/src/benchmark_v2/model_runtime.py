from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .adapters import (
    BaselineScalerContext,
    DLinearAdapter,
    ITransformerAdapter,
    LightTSAdapter,
    NodeSharedGRUAdapter,
    PatchTSTAdapter,
    SegRNNAdapter,
    StatisticalBaselineAdapter,
    TimeXerAdapter,
    TiDEAdapter,
    TransformerAdapter,
    TimesNetAdapter,
    MICNAdapter,
    WPMixerAdapter,
    MultiPatchFormerAdapter,
    TimeMixerAdapter,
    TSMixerAdapter,
    FreTSAdapter,
    CrossformerAdapter,
    MSGNetAdapter,
    TimeFilterAdapter,
    NativeGraphAdapter,
    NativeAdaptiveNodeAdapter,
)
from .configs import (
    resolve_dlinear_config,
    resolve_gru_config,
    resolve_itransformer_config,
    resolve_lightts_config,
    resolve_moving_average_config,
    resolve_patchtst_config,
    resolve_persistence_config,
    resolve_segrnn_config,
    resolve_tide_config,
    resolve_timexer_config,
    resolve_transformer_config,
    resolve_timesnet_config,
    resolve_micn_config,
    resolve_wpmixer_config,
    resolve_multipatchformer_config,
    resolve_timemixer_config,
    resolve_tsmixer_config,
    resolve_frets_config,
    resolve_crossformer_config,
    resolve_msgnet_config,
    resolve_timefilter_config,
    resolve_gcn_config,
    resolve_stgcn_config,
    resolve_dcrnn_config,
    resolve_graph_wavenet_config,
    resolve_mtgnn_config,
    resolve_agcrn_config,
    resolve_stid_config,
)
from .errors import ModelUnavailableError
from .graph import GraphBundle, load_graph_bundle
from .models.graph_models.common import graph_identity_dict
from .models.graph_models.adaptive_common import e3_c_graph_identity
from .registry import load_registry


@dataclass
class ModelRuntime:
    model_id: str
    model: Any
    adapter: Any
    effective_config: dict[str, Any]
    scaler_context: BaselineScalerContext | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    graph_bundle: GraphBundle | None = None

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
    if entry.canonical_id == "persistence":
        config = resolve_persistence_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "moving_average":
        config = resolve_moving_average_config(
            protocol, run_mode=run_mode, ma_window=ma_window
        )
    elif entry.canonical_id == "gru":
        config = resolve_gru_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "dlinear":
        config = resolve_dlinear_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "lightts":
        config = resolve_lightts_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "tide":
        config = resolve_tide_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "segrnn":
        config = resolve_segrnn_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "transformer":
        config = resolve_transformer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "patchtst":
        config = resolve_patchtst_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "itransformer":
        config = resolve_itransformer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "timexer":
        config = resolve_timexer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "timesnet":
        config = resolve_timesnet_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "micn":
        config = resolve_micn_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "wpmixer":
        config = resolve_wpmixer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "multipatchformer":
        config = resolve_multipatchformer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "timemixer":
        config = resolve_timemixer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "tsmixer":
        config = resolve_tsmixer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "frets":
        config = resolve_frets_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "crossformer":
        config = resolve_crossformer_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "msgnet":
        config = resolve_msgnet_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "timefilter":
        config = resolve_timefilter_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "gcn":
        config = resolve_gcn_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "stgcn":
        config = resolve_stgcn_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "dcrnn":
        config = resolve_dcrnn_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "graph_wavenet":
        config = resolve_graph_wavenet_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "mtgnn":
        config = resolve_mtgnn_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "agcrn":
        config = resolve_agcrn_config(protocol, run_mode=run_mode)
    elif entry.canonical_id == "stid":
        config = resolve_stid_config(protocol, run_mode=run_mode)
    else:
        raise ModelUnavailableError(
            f"No E1-A runtime builder exists for {entry.canonical_id}."
        )
    graph_bundle = None
    if entry.canonical_id in {
        "gcn",
        "stgcn",
        "dcrnn",
        "graph_wavenet",
        "mtgnn",
        "agcrn",
        "stid",
    }:
        graph_bundle = load_graph_bundle()
        model = registry.create_model(
            entry.canonical_id,
            config=config,
            protocol=protocol,
            graph_bundle=graph_bundle,
            _allow_blocked_smoke=allow_blocked_smoke,
        )
    else:
        model = registry.create_model(
            entry.canonical_id,
            config=config,
            protocol=protocol,
            _allow_blocked_smoke=allow_blocked_smoke,
        )
    if entry.canonical_id in {
        "gru",
        "dlinear",
        "lightts",
        "tide",
        "segrnn",
        "transformer",
        "patchtst",
        "itransformer",
        "timexer",
        "timesnet",
        "micn",
        "wpmixer",
        "multipatchformer",
        "timemixer",
        "tsmixer",
        "frets",
        "crossformer",
        "msgnet",
        "timefilter",
        "gcn",
        "stgcn",
        "dcrnn",
        "graph_wavenet",
        "mtgnn",
        "agcrn",
        "stid",
    }:
        target_mean, target_std = 0.0, 1.0
        if target_scaler is not None:
            target_mean = float(np.asarray(target_scaler.mean).reshape(-1)[0])
            target_std = float(np.asarray(target_scaler.std).reshape(-1)[0])
        adapter_by_model = {
            "gru": NodeSharedGRUAdapter,
            "dlinear": DLinearAdapter,
            "lightts": LightTSAdapter,
            "tide": TiDEAdapter,
            "segrnn": SegRNNAdapter,
            "transformer": TransformerAdapter,
            "patchtst": PatchTSTAdapter,
            "itransformer": ITransformerAdapter,
            "timexer": TimeXerAdapter,
            "timesnet": TimesNetAdapter,
            "micn": MICNAdapter,
            "wpmixer": WPMixerAdapter,
            "multipatchformer": MultiPatchFormerAdapter,
            "timemixer": TimeMixerAdapter,
            "tsmixer": TSMixerAdapter,
            "frets": FreTSAdapter,
            "crossformer": CrossformerAdapter,
            "msgnet": MSGNetAdapter,
            "timefilter": TimeFilterAdapter,
        }
        if entry.canonical_id in {"gcn", "stgcn", "dcrnn"}:
            assert graph_bundle is not None
            adapter = NativeGraphAdapter(entry.canonical_id, graph_bundle)
            config.update(
                graph_identity_dict(
                    graph_bundle,
                    tuple(config["graph_support_names"]),
                )
            )
            config["source_hash"] = entry.values.get("source_sha256")
            config["source_closure_manifest"] = entry.values.get(
                "source_closure_manifest"
            )
        elif entry.canonical_id in {
            "graph_wavenet",
            "mtgnn",
            "agcrn",
            "stid",
        }:
            assert graph_bundle is not None
            adapter = NativeAdaptiveNodeAdapter(
                entry.canonical_id, graph_bundle
            )
            config.update(
                e3_c_graph_identity(graph_bundle, entry.canonical_id)
            )
            config["source_hash"] = entry.values.get("source_sha256")
            config["source_closure_hash"] = entry.values.get("source_sha256")
            config["source_closure_manifest"] = entry.values.get(
                "source_closure_manifest"
            )
        elif entry.canonical_id == "gru":
            adapter = NodeSharedGRUAdapter()
        else:
            if entry.canonical_id == "transformer":
                adapter = TransformerAdapter(
                    protocol, label_len=int(config["label_len"])
                )
            elif entry.canonical_id == "timesnet":
                adapter = TimesNetAdapter(
                    protocol, top_k=int(config["top_k"])
                )
            else:
                adapter = adapter_by_model[entry.canonical_id](protocol)
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
    context = BaselineScalerContext.from_scalers(
        protocol, input_scaler, target_scaler
    )
    return ModelRuntime(
        model_id=entry.canonical_id,
        model=model,
        adapter=StatisticalBaselineAdapter(context),
        effective_config=config,
        scaler_context=context,
    )
