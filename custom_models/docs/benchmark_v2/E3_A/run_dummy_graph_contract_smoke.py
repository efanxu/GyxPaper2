from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_v2.graph import (  # noqa: E402
    identity_from_bundle,
    load_graph_bundle,
    validate_graph_identity,
    validate_native_graph_input,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    import torch

    bundle = load_graph_bundle()
    identity = identity_from_bundle(bundle)
    x = torch.zeros(2, 144, 134, 16, dtype=torch.float32)
    target = torch.full((2, 134, 10), 999.0)
    mask = torch.zeros(2, 134, 10, dtype=torch.bool)
    validate_native_graph_input(
        x,
        runtime_node_ids=list(bundle.ordered_node_ids),
        bundle=bundle,
    )

    class DummyGraphContractOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_argument_count = None

        def forward(self, value, graph_bundle):
            self.forward_argument_count = 2
            graph_bundle.validate_runtime_identity(
                graph_protocol_hash=identity.graph_protocol_hash,
                node_order_hash=identity.node_order_hash,
                graph_bundle_hash=identity.graph_bundle_hash,
            )
            return value[:, -1, :, 15:16].repeat(1, 1, 10)

    model = DummyGraphContractOnly()
    prediction = model(x, bundle)
    if tuple(prediction.shape) != (2, 134, 10):
        raise RuntimeError(f"Unexpected dummy shape: {prediction.shape}")
    if model.forward_argument_count != 2:
        raise RuntimeError("Dummy forward received unexpected arguments.")
    artifact = {
        "schema_version": "e3_a_dummy_graph_contract_v1",
        "formal_registry_entry": False,
        "benchmark_result": False,
        "dummy_contract_only": True,
        "input_shape": [2, 144, 134, 16],
        "prediction_shape": [2, 134, 10],
        "target_shape": list(target.shape),
        "mask_shape": list(mask.shape),
        "target_passed_to_forward": False,
        "mask_passed_to_forward": False,
        **identity.to_dict(),
    }
    validate_graph_identity(
        identity, artifact, context="dummy strict reload"
    )
    smoke_root = (
        PROJECT_ROOT
        / "custom_models/results_smoke/benchmark_v2/e3_a_contract"
    )
    write_json(smoke_root / "dummy_graph_identity.json", artifact)
    checkpoint_metadata = {
        "schema_version": "e3_a_dummy_checkpoint_metadata_v1",
        "model_id": "__e3_a_dummy_graph_contract_only__",
        **identity.to_dict(),
    }
    validate_graph_identity(
        identity,
        checkpoint_metadata,
        context="dummy checkpoint strict reload",
    )
    write_json(
        smoke_root / "dummy_checkpoint_metadata.json", checkpoint_metadata
    )
    result = {
        "status": "PASS",
        "cpu_only": True,
        "cuda_used": False,
        "model_registered": False,
        "formal_result_created": False,
        "strict_reload_graph_identity": "PASS",
        **artifact,
    }
    write_json(
        Path(__file__).resolve().parent
        / "dummy_graph_contract_smoke_results.json",
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
