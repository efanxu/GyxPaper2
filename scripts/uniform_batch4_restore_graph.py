from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
sys.path.insert(0, str(SRC_ROOT))

from st_mgprompt.config import STMGPromptConfig
from st_mgprompt.data import make_dataloaders
from st_mgprompt.graph_prior import build_graph_artifacts


EXPECTED = {
    "macro_trend_adjacency.npy": (
        "1ec6a0e06156f7a9a6b750b239e8f3dd21ade09aa354a006d18f5df59d677ff4"
    ),
    "micro_local_adjacency.npy": (
        "415931db3133b844c693a20e317f5b5facb5d57305eed8eeb2639f6ebc498f9d"
    ),
    "metadata.json": (
        "c3b0a90d1e73c39b14058d0405ecb42f547f233fed5e4b89d18e117669eb4cd2"
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def config() -> STMGPromptConfig:
    cfg = STMGPromptConfig()
    cfg.smoke = False
    cfg.smoke_use_synthetic = False
    cfg.model_input_path = "dataset/sdwpf_model_input_base.parquet"
    cfg.eval_target_path = "dataset/sdwpf_eval_target.parquet"
    cfg.location_path = "dataset/sdwpf_turb_location_elevation.csv"
    cfg.num_nodes = 134
    cfg.num_workers = 0
    cfg.train_batch_size = 4
    cfg.val_batch_size = 4
    cfg.test_batch_size = 4
    cfg.batch_size = 4
    cfg.eval_batch_size = 4
    cfg.use_adaptive_graph = True
    return cfg


def main() -> int:
    cfg = config()
    target = (
        PROJECT_ROOT
        / "custom_models"
        / "graphs"
        / "trend_prior_v4_2"
    )
    adjacency_names = (
        "macro_trend_adjacency.npy",
        "micro_local_adjacency.npy",
    )
    if all(
        (target / name).is_file()
        and digest(target / name) == EXPECTED[name]
        for name in adjacency_names
    ):
        metadata_path = target / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("num_nodes") != 134:
            raise RuntimeError("Refusing metadata repair for a non-134-node graph")
        metadata["uses_adaptive_graph"] = True
        metadata["adaptive_graph_constrained_by_prior"] = True
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        actual = {name: digest(target / name) for name in EXPECTED}
        if actual != EXPECTED:
            raise RuntimeError(
                "Protected metadata repair did not reproduce frozen hashes: "
                + json.dumps(actual, sort_keys=True)
            )
        print(
            json.dumps(
                {
                    "status": "RESTORED",
                    "target": str(target),
                    "hashes": actual,
                    "repair_mode": "metadata_after_verified_adjacency_restore",
                    "pid": os.getpid(),
                },
                ensure_ascii=False,
            )
        )
        return 0

    bundle = make_dataloaders(cfg)["bundle"]
    probe_root = (
        PROJECT_ROOT
        / "custom_models"
        / "results_smoke"
        / "uniform_batch4_graph_restore_probe"
    )
    cfg.graph_output_root = str(probe_root)
    cfg.graph_tag = "trend_prior_v4_2_probe"
    build_graph_artifacts(bundle, cfg)
    probe = probe_root / cfg.graph_tag
    probe_hashes = {
        name: digest(probe / name)
        for name in ("macro_trend_adjacency.npy", "micro_local_adjacency.npy")
    }
    for name, actual in probe_hashes.items():
        if actual != EXPECTED[name]:
            raise RuntimeError(
                f"Refusing protected graph restore: {name} probe hash "
                f"{actual} != {EXPECTED[name]}"
            )

    cfg.graph_output_root = str(PROJECT_ROOT / "custom_models" / "graphs")
    cfg.graph_tag = "trend_prior_v4_2"
    build_graph_artifacts(bundle, cfg)
    target = PROJECT_ROOT / "custom_models" / "graphs" / cfg.graph_tag
    actual = {name: digest(target / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(
            "Protected graph regeneration did not reproduce the frozen hashes: "
            + json.dumps(actual, sort_keys=True)
        )
    print(
        json.dumps(
            {
                "status": "RESTORED",
                "target": str(target),
                "hashes": actual,
                "probe_preserved": str(probe),
                "pid": os.getpid(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
