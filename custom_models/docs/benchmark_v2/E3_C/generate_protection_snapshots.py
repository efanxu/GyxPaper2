from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
FORMAL_OUTPUT_ROOT = (
    PROJECT_ROOT / "custom_models/results/benchmark_v2/e3_c_seed2026"
)

GROUPS: dict[str, list[str]] = {
    "core": [
        "custom_models/src/st_mgprompt",
        "custom_models/results/st_mgprompt_canonical",
        "custom_models/results/st_mgprompt_precision",
        "custom_models/results/st_mgprompt_component_ablation",
        "custom_models/graphs/trend_prior_v4_2/metadata.json",
    ],
    "tslib": [
        "Time-Series-Library/models",
        "Time-Series-Library/layers",
        "Time-Series-Library/exp",
        "Time-Series-Library/data_provider",
        "Time-Series-Library/run.py",
        "Time-Series-Library/README.md",
        "Time-Series-Library/LICENSE",
        "Time-Series-Library/scripts",
    ],
    "benchmark_frozen": [
        "custom_models/src/benchmark_v2/protocol/benchmark_protocol_v1.json",
        "custom_models/src/benchmark_v2/data",
        "custom_models/src/benchmark_v2/schemas",
        "custom_models/src/benchmark_v2/contracts",
        "custom_models/src/benchmark_v2/losses.py",
        "custom_models/src/benchmark_v2/metrics.py",
        "custom_models/src/benchmark_v2/checkpointing.py",
        "custom_models/src/benchmark_v2/artifacts.py",
        "custom_models/src/benchmark_v2/engine.py",
        "custom_models/src/benchmark_v2/hardware_preflight.py",
        "custom_models/src/benchmark_v2/model_runtime.py",
        "custom_models/src/benchmark_v2/model_cli.py",
        "custom_models/src/benchmark_v2/cli.py",
        "custom_models/src/benchmark_v2/runtime.py",
        "custom_models/src/benchmark_v2/run_benchmark.py",
        "custom_models/src/benchmark_v2/registry",
        "custom_models/src/benchmark_v2/upstream/tslib_loader.py",
    ],
    "graph_protocol": [
        "custom_models/src/benchmark_v2/protocol/graph_protocol_v1.json",
        "custom_models/src/benchmark_v2/protocol/graph_v1",
        "custom_models/src/benchmark_v2/graph",
    ],
    "e1_a": [
        "custom_models/src/benchmark_v2/models/persistence.py",
        "custom_models/src/benchmark_v2/models/moving_average.py",
        "custom_models/src/benchmark_v2/models/gru.py",
        "custom_models/src/benchmark_v2/configs/persistence.py",
        "custom_models/src/benchmark_v2/configs/moving_average.py",
        "custom_models/src/benchmark_v2/configs/gru.py",
        "custom_models/src/benchmark_v2/adapters/statistical_baselines.py",
        "custom_models/src/benchmark_v2/adapters/gru.py",
        "custom_models/tests/benchmark_v2/test_e1_a_registry_cli.py",
        "custom_models/docs/benchmark_v2/E1_A",
    ],
    "e1_b": [
        "custom_models/src/benchmark_v2/models/tide.py",
        "custom_models/src/benchmark_v2/models/segrnn.py",
        "custom_models/src/benchmark_v2/configs/tide.py",
        "custom_models/src/benchmark_v2/configs/segrnn.py",
        "custom_models/src/benchmark_v2/adapters/tide.py",
        "custom_models/src/benchmark_v2/adapters/segrnn.py",
        "custom_models/src/benchmark_v2/adapters/tslib_channel.py",
        "custom_models/tests/benchmark_v2/test_e1_b_registry_cli.py",
        "custom_models/tests/benchmark_v2/test_e1_b_loader.py",
        "custom_models/docs/benchmark_v2/E1_B",
    ],
    "e2_a": [
        "custom_models/src/benchmark_v2/models/transformer.py",
        "custom_models/src/benchmark_v2/models/patchtst.py",
        "custom_models/src/benchmark_v2/models/itransformer.py",
        "custom_models/src/benchmark_v2/models/timexer.py",
        "custom_models/src/benchmark_v2/configs/transformer.py",
        "custom_models/src/benchmark_v2/configs/patchtst.py",
        "custom_models/src/benchmark_v2/configs/itransformer.py",
        "custom_models/src/benchmark_v2/configs/timexer.py",
        "custom_models/src/benchmark_v2/adapters/e2_a.py",
        "custom_models/tests/benchmark_v2/test_e2_a_models.py",
        "custom_models/docs/benchmark_v2/E2_A",
    ],
    "e2_b": [
        "custom_models/src/benchmark_v2/models/timesnet.py",
        "custom_models/src/benchmark_v2/models/micn.py",
        "custom_models/src/benchmark_v2/models/wpmixer.py",
        "custom_models/src/benchmark_v2/models/multipatchformer.py",
        "custom_models/src/benchmark_v2/configs/timesnet.py",
        "custom_models/src/benchmark_v2/configs/micn.py",
        "custom_models/src/benchmark_v2/configs/wpmixer.py",
        "custom_models/src/benchmark_v2/configs/multipatchformer.py",
        "custom_models/src/benchmark_v2/adapters/e2_b.py",
        "custom_models/tests/benchmark_v2/test_e2_b_models.py",
        "custom_models/docs/benchmark_v2/E2_B",
    ],
    "e2_c": [
        "custom_models/src/benchmark_v2/models/timemixer.py",
        "custom_models/src/benchmark_v2/models/tsmixer.py",
        "custom_models/src/benchmark_v2/models/frets.py",
        "custom_models/src/benchmark_v2/configs/timemixer.py",
        "custom_models/src/benchmark_v2/configs/tsmixer.py",
        "custom_models/src/benchmark_v2/configs/frets.py",
        "custom_models/src/benchmark_v2/adapters/e2_c.py",
        "custom_models/tests/benchmark_v2/test_e2_c_models.py",
        "custom_models/docs/benchmark_v2/E2_C",
    ],
    "e2_d": [
        "custom_models/src/benchmark_v2/models/crossformer.py",
        "custom_models/src/benchmark_v2/models/msgnet.py",
        "custom_models/src/benchmark_v2/models/timefilter.py",
        "custom_models/src/benchmark_v2/configs/crossformer.py",
        "custom_models/src/benchmark_v2/configs/msgnet.py",
        "custom_models/src/benchmark_v2/configs/timefilter.py",
        "custom_models/src/benchmark_v2/adapters/e2_d.py",
        "custom_models/tests/benchmark_v2/test_e2_d_models.py",
        "custom_models/docs/benchmark_v2/E2_D",
    ],
    "e3_a": [
        "custom_models/tests/benchmark_v2/test_e3_a_graph_protocol.py",
        "custom_models/docs/benchmark_v2/E3_A",
    ],
    "e3_b": [
        "custom_models/src/benchmark_v2/models/graph_models",
        "custom_models/src/benchmark_v2/configs/graph_common.py",
        "custom_models/src/benchmark_v2/configs/gcn.py",
        "custom_models/src/benchmark_v2/configs/stgcn.py",
        "custom_models/src/benchmark_v2/configs/dcrnn.py",
        "custom_models/src/benchmark_v2/adapters/e3_b.py",
        "custom_models/tests/benchmark_v2/test_e3_b_graph_models.py",
        "custom_models/docs/benchmark_v2/E3_B",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_path(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return {"path": relative_path, "exists": False}
    if path.is_file():
        return {
            "path": relative_path,
            "exists": True,
            "kind": "file",
            "sha256": sha256_file(path),
        }
    records = [
        (item.relative_to(path).as_posix(), sha256_file(item))
        for item in sorted(path.rglob("*"))
        if item.is_file() and "__pycache__" not in item.parts
    ]
    material = "".join(f"{name}:{digest}\n" for name, digest in records)
    return {
        "path": relative_path,
        "exists": True,
        "kind": "directory",
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
    }


def make_snapshot(name: str, phase: str, paths: list[str]) -> dict[str, Any]:
    protocol_path = (
        PROJECT_ROOT
        / "custom_models/src/benchmark_v2/protocol/benchmark_protocol_v1.json"
    )
    canonical_path = (
        PROJECT_ROOT
        / "custom_models/results/st_mgprompt_canonical/"
        "full_fixed_dual_keep_msmgdwu_seed2026/best_checkpoint.pt"
    )
    payload: dict[str, Any] = {
        "task": "E3-C",
        "snapshot": f"E3-C task {phase}",
        "algorithm": (
            "SHA256; directory digest is SHA256 of sorted "
            "relative_path:sha256 newline records; __pycache__ excluded"
        ),
        "paths": [snapshot_path(path) for path in paths],
        "protocol_file_sha256": sha256_file(protocol_path),
        "canonical_checkpoint_sha256": sha256_file(canonical_path),
        "formal_output_root_existed": FORMAL_OUTPUT_ROOT.exists(),
    }
    if phase == "after":
        before = json.loads(
            (
                OUTPUT_DIR / f"protected_{name}_hashes_before.json"
            ).read_text(encoding="utf-8")
        )
        before_map = {entry["path"]: entry for entry in before["paths"]}
        mismatches: list[str] = []
        for entry in payload["paths"]:
            entry["matches_before"] = entry == before_map.get(entry["path"])
            if not entry["matches_before"]:
                mismatches.append(entry["path"])
        payload["mismatches"] = mismatches
        payload["protocol_matches_before"] = (
            payload["protocol_file_sha256"] == before["protocol_file_sha256"]
        )
        payload["canonical_matches_before"] = (
            payload["canonical_checkpoint_sha256"]
            == before["canonical_checkpoint_sha256"]
        )
        payload["formal_output_root_matches_before"] = (
            payload["formal_output_root_existed"]
            == before["formal_output_root_existed"]
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    for name, paths in GROUPS.items():
        output = OUTPUT_DIR / f"protected_{name}_hashes_{args.phase}.json"
        output.write_text(
            json.dumps(
                make_snapshot(name, args.phase, paths),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(output.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()
