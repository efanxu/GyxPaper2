from __future__ import annotations

from typing import Any

from .constants import ARCHITECTURE_FAMILIES, DISPLAY_NAMES, E9_SCOPE_ID, LOSS_ID, MODEL_IDS

CONTROL_RUN_IDS = {
    "lightts": "LightTS_node_shared_chunk8_bs4_seed2026",
    "tide": "TiDE_node_shared_d512_bs4_seed2026",
    "patchtst": "PatchTST_node_shared_p16_s8_bs4_seed2026",
    "itransformer": "iTransformer_node_shared_d512_bs4_seed2026",
    "dcrnn": "DCRNN_native_dualrw_k2_h64_l2_ar_bs4_seed2026",
    "mtgnn": "MTGNN_native_adaptive_k20_gdep2_l3_bs4_seed2026",
}


def canonical_transfer_run_id(model_id: str) -> str:
    control = CONTROL_RUN_IDS[model_id]
    suffix = "_seed2026"
    if not control.endswith(suffix):
        raise ValueError(f"Control run-id does not follow the frozen seed suffix: {control}")
    return control[: -len(suffix)] + "_msmg_dwu_seed2026"


def variant_manifest() -> dict[str, Any]:
    variants = [
        {
            "variant_id": f"E9_{model_id.upper()}_MSMG_DWU",
            "scope_id": E9_SCOPE_ID,
            "model_id": model_id,
            "display_name": DISPLAY_NAMES[model_id],
            "architecture_family": ARCHITECTURE_FAMILIES[model_id],
            "control_run_id": CONTROL_RUN_IDS[model_id],
            "transfer_run_id": canonical_transfer_run_id(model_id),
            "loss_id": LOSS_ID,
            "formal_training": True,
            "control_training": False,
        }
        for model_id in MODEL_IDS
    ]
    return {
        "schema_version": "e9_variant_manifest_v1",
        "scope_id": E9_SCOPE_ID,
        "experiment": "E9-B 6-model loss-only transfer",
        "models_frozen_before_results": True,
        "variant_count": len(variants),
        "variants": variants,
        "excluded_models": [
            "persistence", "gru", "dlinear", "crossformer", "graph_wavenet", "agcrn", "segrnn", "msgnet",
        ],
        "e5_consumed": False,
    }


def run_id_map() -> dict[str, Any]:
    return {
        "schema_version": "e9_run_id_map_v1",
        "scope_id": E9_SCOPE_ID,
        "runs": {
            model_id: {
                "control": CONTROL_RUN_IDS[model_id],
                "transfer": canonical_transfer_run_id(model_id),
            }
            for model_id in MODEL_IDS
        },
    }
