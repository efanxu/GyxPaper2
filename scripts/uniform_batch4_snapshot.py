from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "docs"
    / "benchmark_v2"
    / "UNIFORM_BATCH4_PROTOCOL"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def file_records(paths: Iterable[Path], *, include_hash: bool = True) -> list[dict]:
    records: list[dict] = []
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            records.append({"path": relative(path), "exists": False})
            continue
        stat = path.stat()
        record = {
            "path": relative(path),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if include_hash:
            record["sha256"] = sha256(path)
        records.append(record)
    return records


def write(name: str, phase: str, files: list[dict], **extra) -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "uniform_batch4_protection_snapshot_v1",
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "files": files,
        **extra,
    }
    (DOC_ROOT / f"{name}_{phase}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def registry_sources() -> tuple[list[Path], list[Path]]:
    registry_path = (
        PROJECT_ROOT
        / "custom_models"
        / "src"
        / "benchmark_v2"
        / "registry"
        / "benchmark_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry["models"] if isinstance(registry, dict) else registry
    sources: list[Path] = []
    tslib: list[Path] = []
    for entry in entries:
        source = PROJECT_ROOT / entry["source_path"]
        sources.append(source)
        if entry.get("source_type") == "tslib_source":
            tslib.append(source)
    return sources, tslib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    phase = args.phase

    benchmark_root = PROJECT_ROOT / "custom_models" / "src" / "benchmark_v2"
    protocol_root = benchmark_root / "protocol"
    results_root = PROJECT_ROOT / "custom_models" / "results"
    model_sources, tslib_sources = registry_sources()

    canonical = (
        results_root
        / "st_mgprompt_canonical"
        / "full_fixed_dual_keep_msmgdwu_seed2026"
        / "best_checkpoint.pt"
    )
    core = [
        protocol_root / "benchmark_protocol_v1.json",
        protocol_root / "graph_protocol_v1.json",
        protocol_root / "graph_v1" / "graph_bundle_manifest_v1.json",
        protocol_root / "graph_v1" / "node_order_v1.json",
        benchmark_root
        / "experiments"
        / "e5_common_loss"
        / "e5_common_loss_protocol_v1.json",
        canonical,
    ]
    write(
        "protected_core_hashes",
        phase,
        file_records(core),
        expected_benchmark_protocol_hash=(
            "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b"
        ),
        expected_canonical_checkpoint_sha256=(
            "f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a"
        ),
    )
    write("protected_model_sources", phase, file_records(model_sources))
    write(
        "protected_model_configs",
        phase,
        file_records((benchmark_root / "configs").glob("*.py")),
    )
    write(
        "protected_tslib_hashes",
        phase,
        file_records(tslib_sources),
        allowlist_count=len(tslib_sources),
    )
    graph_paths = list(protocol_root.glob("graph_protocol_v1.json"))
    graph_paths.extend((protocol_root / "graph_v1").glob("*"))
    graph_paths.extend(
        path
        for path in (PROJECT_ROOT / "custom_models" / "graphs").rglob("*")
        if path.is_file()
    )
    write("protected_graph_hashes", phase, file_records(graph_paths))

    st_root = PROJECT_ROOT / "custom_models" / "src" / "st_mgprompt"
    st_architecture_names = {
        "__init__.py",
        "config.py",
        "coupling_block.py",
        "cross_fusion.py",
        "decoder.py",
        "graph_layers.py",
        "graph_prior.py",
        "losses.py",
        "model.py",
        "prompt_alignment.py",
        "temporal_layers.py",
        "volatility_patching.py",
    }
    write(
        "protected_st_mgprompt_hashes",
        phase,
        file_records(st_root / name for name in st_architecture_names),
        scope="architecture_config_loss_only",
    )

    a0_a8_root = (
        results_root
        / "st_mgprompt_component_ablation"
        / "component_ablation_fixed_dual_seed2026"
    )
    a0_a8_paths = [
        path
        for path in a0_a8_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".json", ".csv", ".md"}
            or path.name == "best_checkpoint.pt"
        )
    ]
    write("protected_a0_a8_hashes", phase, file_records(a0_a8_paths))

    e5_paths = [
        benchmark_root
        / "experiments"
        / "e5_common_loss"
        / "e5_common_loss_protocol_v1.json"
    ]
    e5_result_root = results_root / "benchmark_v2" / "common_loss_architecture_seed2026"
    if e5_result_root.is_dir():
        e5_paths.extend(path for path in e5_result_root.rglob("*") if path.is_file())
    write("protected_e5_a_hashes", phase, file_records(e5_paths))

    result_files = [
        path for path in results_root.rglob("*") if path.is_file()
    ]
    write(
        "protected_formal_results_inventory",
        phase,
        file_records(result_files, include_hash=False),
        file_count=len(result_files),
        note="Inventory uses path/size/mtime; key protected artifacts are SHA256-covered separately.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
