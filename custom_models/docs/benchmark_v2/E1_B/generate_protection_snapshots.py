from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent

CORE_PATHS = [
    "custom_models/src/st_mgprompt",
    "custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026",
    "custom_models/results/st_mgprompt_precision/precision_ablation_fixed_dual_seed2026",
    "custom_models/results/st_mgprompt_component_ablation/component_ablation_fixed_dual_seed2026",
    "custom_models/graphs/trend_prior_v4_2/metadata.json",
]
TSLIB_PATHS = [
    "Time-Series-Library/models",
    "Time-Series-Library/layers",
    "Time-Series-Library/exp",
    "Time-Series-Library/data_provider",
    "Time-Series-Library/run.py",
    "Time-Series-Library/LICENSE",
]
BENCHMARK_FROZEN_PATHS = [
    "custom_models/src/benchmark_v2/protocol",
    "custom_models/src/benchmark_v2/data",
    "custom_models/src/benchmark_v2/schemas",
    "custom_models/src/benchmark_v2/losses.py",
    "custom_models/src/benchmark_v2/metrics.py",
    "custom_models/src/benchmark_v2/checkpointing.py",
    "custom_models/src/benchmark_v2/artifacts.py",
]
E1_A_PATHS = [
    "custom_models/src/benchmark_v2/models/persistence.py",
    "custom_models/src/benchmark_v2/models/moving_average.py",
    "custom_models/src/benchmark_v2/models/gru.py",
    "custom_models/src/benchmark_v2/configs/persistence.py",
    "custom_models/src/benchmark_v2/configs/moving_average.py",
    "custom_models/src/benchmark_v2/configs/gru.py",
    "custom_models/src/benchmark_v2/adapters/statistical_baselines.py",
    "custom_models/src/benchmark_v2/adapters/gru.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and "__pycache__" not in candidate.parts
    )


def snapshot_path(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return {"path": relative_path, "exists": False}
    files = files_under(path)
    if path.is_file():
        return {
            "path": relative_path,
            "exists": True,
            "kind": "file",
            "sha256": sha256_file(path),
        }
    records = [
        (
            file.relative_to(path).as_posix(),
            sha256_file(file),
        )
        for file in files
    ]
    tree_material = "".join(f"{name}:{digest}\n" for name, digest in records)
    return {
        "path": relative_path,
        "exists": True,
        "kind": "directory",
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(tree_material.encode("utf-8")).hexdigest(),
    }


def snapshot(name: str, phase: str, paths: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "E1-B",
        "snapshot": f"E1-B task {phase}",
        "algorithm": (
            "SHA256; directory digest is SHA256 of sorted "
            "relative_path:sha256 newline records; __pycache__ excluded"
        ),
        "paths": [snapshot_path(path) for path in paths],
    }
    if name == "core":
        checkpoint = (
            PROJECT_ROOT
            / "custom_models/results/st_mgprompt_canonical/"
            "full_fixed_dual_keep_msmgdwu_seed2026/best_checkpoint.pt"
        )
        payload["canonical_checkpoint_sha256"] = (
            sha256_file(checkpoint) if checkpoint.is_file() else None
        )
        payload["formal_output_root_existed"] = (
            PROJECT_ROOT / "custom_models/results/benchmark_v2"
        ).exists()
    if phase == "after":
        before_path = OUTPUT_DIR / f"protected_{name}_hashes_before.json"
        before = json.loads(before_path.read_text(encoding="utf-8"))
        before_map = {entry["path"]: entry for entry in before["paths"]}
        mismatches = []
        for entry in payload["paths"]:
            previous = before_map.get(entry["path"])
            entry["matches_before"] = entry == previous
            if not entry["matches_before"]:
                mismatches.append(entry["path"])
        payload["mismatches"] = mismatches
        if name == "core":
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
    groups = {
        "core": CORE_PATHS,
        "tslib": TSLIB_PATHS,
        "benchmark_frozen": BENCHMARK_FROZEN_PATHS,
        "e1_a": E1_A_PATHS,
    }
    for name, paths in groups.items():
        payload = snapshot(name, args.phase, paths)
        output_path = OUTPUT_DIR / f"protected_{name}_hashes_{args.phase}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(output_path.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()
