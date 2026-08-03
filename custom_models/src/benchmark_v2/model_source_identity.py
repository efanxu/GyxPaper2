from __future__ import annotations

import hashlib
import json
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_SOURCE_IDENTITY_SCHEMA_VERSION = "model_source_identity_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelSourceIdentityError(ValueError):
    """Raised when a model source closure cannot be audited deterministically."""


def normalize_relative_path(value: str | Path) -> str:
    """Return a repository-relative POSIX path or fail closed."""

    raw = str(value).replace("\\", "/")
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raise ModelSourceIdentityError(
            f"Absolute model source path is not allowed in identity material: {value!r}"
        )
    normalized = posixpath.normpath(raw)
    if (
        normalized in {"", ".", ".."}
        or normalized.startswith("../")
        or PurePosixPath(normalized).is_absolute()
    ):
        raise ModelSourceIdentityError(
            f"Invalid repository-relative model source path: {value!r}"
        )
    return normalized


def _sha256_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ModelSourceIdentityError(
            f"Model source closure is not valid UTF-8: {path}"
        ) from exc
    canonical_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _resolve_repo_file(relative_path: str, project_root: Path) -> Path:
    root = project_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ModelSourceIdentityError(
            f"Model source path escapes the repository: {relative_path}"
        ) from exc
    if not candidate.is_file():
        raise ModelSourceIdentityError(
            f"Model source closure file is missing: {relative_path}"
        )
    return candidate


def canonical_hash_for_records(
    records: Iterable[Mapping[str, Any]],
    *,
    schema_version: str = MODEL_SOURCE_IDENTITY_SCHEMA_VERSION,
) -> str:
    """Hash path/content records using platform-independent canonical JSON."""

    canonical: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        path = normalize_relative_path(str(record.get("path", "")))
        if path in seen:
            raise ModelSourceIdentityError(
                f"Duplicate model source closure path: {path}"
            )
        seen.add(path)
        file_hash = str(record.get("sha256", "")).casefold()
        if not _SHA256_RE.fullmatch(file_hash):
            raise ModelSourceIdentityError(
                f"Invalid SHA256 for model source closure file: {path}"
            )
        canonical.append({"path": path, "sha256": file_hash})
    if not canonical:
        raise ModelSourceIdentityError("Model source closure cannot be empty.")
    canonical.sort(key=lambda item: item["path"])
    material = json.dumps(
        {"schema_version": schema_version, "files": canonical},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _module_path(module_name: str) -> str:
    module = module_name.split(":", 1)[0]
    if module.startswith("benchmark_v2."):
        return "custom_models/src/" + module.replace(".", "/") + ".py"
    return module.replace(".", "/") + ".py"


def _config_path(config_schema: str) -> str:
    value = str(config_schema).split(":", 1)[0]
    if value.endswith(".py"):
        return value
    return _module_path(value)


def _adapter_path(model_id: str) -> str:
    if model_id in {"persistence", "moving_average"}:
        return "custom_models/src/benchmark_v2/adapters/statistical_baselines.py"
    if model_id == "gru":
        return "custom_models/src/benchmark_v2/adapters/gru.py"
    if model_id == "dlinear":
        return "custom_models/src/benchmark_v2/adapters/dlinear.py"
    if model_id == "lightts":
        return "custom_models/src/benchmark_v2/adapters/lightts.py"
    if model_id == "tide":
        return "custom_models/src/benchmark_v2/adapters/tide.py"
    if model_id == "segrnn":
        return "custom_models/src/benchmark_v2/adapters/segrnn.py"
    if model_id in {"transformer", "patchtst", "itransformer", "timexer"}:
        return "custom_models/src/benchmark_v2/adapters/e2_a.py"
    if model_id in {"timesnet", "micn", "wpmixer", "multipatchformer"}:
        return "custom_models/src/benchmark_v2/adapters/e2_b.py"
    if model_id in {"timemixer", "tsmixer", "frets"}:
        return "custom_models/src/benchmark_v2/adapters/e2_c.py"
    if model_id in {"crossformer", "msgnet", "timefilter"}:
        return "custom_models/src/benchmark_v2/adapters/e2_d.py"
    if model_id in {"gcn", "stgcn", "dcrnn"}:
        return "custom_models/src/benchmark_v2/adapters/e3_b.py"
    if model_id in {"graph_wavenet", "mtgnn", "agcrn", "stid"}:
        return "custom_models/src/benchmark_v2/adapters/e3_c.py"
    raise ModelSourceIdentityError(f"No audited adapter closure for {model_id!r}")


def _closure_paths(model_id: str, entry: Any) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []

    def add(path: str, role: str) -> None:
        normalized = normalize_relative_path(path)
        if normalized not in {item[0] for item in paths}:
            paths.append((normalized, role))

    add(str(entry.source_path), "model implementation source")
    add(_module_path(str(entry.implementation_module)), "benchmark model wrapper")
    add(_config_path(str(entry.config_schema)), "model-specific config resolver")
    add(_adapter_path(model_id), "benchmark adapter implementation")
    add(
        "custom_models/src/benchmark_v2/model_runtime.py",
        "explicit model runtime builder",
    )
    add(
        "custom_models/src/benchmark_v2/adapters/node_shared.py",
        "shared adapter contract used by the model adapter",
    )
    if str(entry.source_type) == "tslib_source":
        add(
            "custom_models/src/benchmark_v2/upstream/tslib_loader.py",
            "audited TSLib loader and source resolver",
        )
    if model_id in {"persistence", "moving_average"}:
        add(
            "custom_models/src/benchmark_v2/adapters/statistical_baselines.py",
            "baseline scaler adapter implementation",
        )
    if model_id in {"gcn", "stgcn", "dcrnn"}:
        add(
            "custom_models/src/benchmark_v2/models/graph_models/common.py",
            "native graph support and identity boundary",
        )
        add(
            "custom_models/src/benchmark_v2/models/graph_models/__init__.py",
            "native graph model exports",
        )
    if model_id in {"graph_wavenet", "mtgnn", "agcrn", "stid"}:
        add(
            "custom_models/src/benchmark_v2/models/graph_models/adaptive_common.py",
            "adaptive graph identity and diagnostics",
        )
    return paths


def canonical_model_source_identity(model_id: str) -> dict[str, Any]:
    registry_entry = load_registry().get(model_id)
    paths = _closure_paths(registry_entry.canonical_id, registry_entry)
    records: list[dict[str, str]] = []
    for relative_path, role in paths:
        resolved = _resolve_repo_file(relative_path, PROJECT_ROOT)
        records.append(
            {
                "path": relative_path,
                "sha256": _sha256_file(resolved),
                "role": role,
            }
        )
    records.sort(key=lambda item: item["path"])
    combined_hash = canonical_hash_for_records(records)
    return {
        "source_identity_schema_version": MODEL_SOURCE_IDENTITY_SCHEMA_VERSION,
        "model_id": registry_entry.canonical_id,
        "source_closure_files": records,
        "canonical_combined_hash": combined_hash,
    }


def canonical_model_source_hash(model_id: str) -> str:
    return str(canonical_model_source_identity(model_id)["canonical_combined_hash"])


def canonical_model_source_closure_manifest(model_id: str) -> dict[str, Any]:
    return canonical_model_source_identity(model_id)
