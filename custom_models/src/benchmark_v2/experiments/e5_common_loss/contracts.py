from __future__ import annotations

TRAINABLE_MODELS = (
    "gru",
    "dlinear",
    "lightts",
    "tide",
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
    "timefilter",
    "gcn",
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "mtgnn",
    "agcrn",
    "stid",
)
NONTRAINABLE_MODELS = ("persistence", "moving_average")
A8_REFERENCE_ID = "STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference"
BATCH4_A8_REFERENCE_ID = A8_REFERENCE_ID
EXPECTED_BENCHMARK_ENTRIES = 26
EXPECTED_TOTAL_ENTRIES = 27
LEGACY_FORMAL_OUTPUT_ROOT_RELATIVE = (
    "custom_models/results/benchmark_v2/common_loss_architecture_seed2026"
)
SMOKE_OUTPUT_ROOT_RELATIVE = (
    "custom_models/results_smoke/benchmark_v2/e5_common_loss"
)
BATCH4_FORMAL_OUTPUT_ROOT_RELATIVE = (
    "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026"
)
FORMAL_OUTPUT_ROOT_RELATIVE = BATCH4_FORMAL_OUTPUT_ROOT_RELATIVE
BATCH4_SMOKE_OUTPUT_ROOT_RELATIVE = (
    "custom_models/results_smoke/benchmark_v2_uniform_bs4/e5_common_loss"
)


def validate_registry_contract(registry) -> dict[str, object]:
    ids = tuple(entry.canonical_id for entry in registry.list())
    expected = NONTRAINABLE_MODELS + TRAINABLE_MODELS
    excluded = {"segrnn", "msgnet"}
    missing = sorted(set(expected) - set(ids))
    unexpected = sorted(set(ids) - set(expected) - excluded)
    trainability_mismatches = []
    for model_id in TRAINABLE_MODELS:
        entry = registry.get(model_id)
        if not entry.supports_train or entry.supports_non_trainable:
            trainability_mismatches.append(model_id)
    for model_id in NONTRAINABLE_MODELS:
        entry = registry.get(model_id)
        if entry.supports_train or not entry.supports_non_trainable:
            trainability_mismatches.append(model_id)
    passed = (
        len(set(ids) & set(expected)) == EXPECTED_BENCHMARK_ENTRIES
        and not missing
        and not unexpected
        and not trainability_mismatches
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "registry_count": len(set(ids) & set(expected)),
        "expected_registry_count": EXPECTED_BENCHMARK_ENTRIES,
        "trainable_count": len(TRAINABLE_MODELS),
        "nontrainable_count": len(NONTRAINABLE_MODELS),
        "a8_reference_count": 1,
        "expected_total_entries": EXPECTED_TOTAL_ENTRIES,
        "missing": missing,
        "unexpected": unexpected,
        "trainability_mismatches": sorted(trainability_mismatches),
    }
