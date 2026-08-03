from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .model_factories.registry import factory_module_path
from .registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_SOURCE_IDENTITY_SCHEMA_VERSION = "model_source_identity_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_LOCAL_IMPORT_ROOTS = {
    "benchmark_v2",
    "custom_models",
    "layers",
    "models",
    "utils",
    "data_provider",
    "exp",
}
_DYNAMIC_IMPORT_ALLOWLIST = {
    # Registry factories are declared by the audited registry JSON and are
    # resolved at runtime; the registry module itself is part of every model
    # closure, while its dynamic target is audited by the registry contract.
    "custom_models/src/benchmark_v2/registry/registry.py",
    # The factory module is a closed literal id -> module allowlist.  Its
    # target is added explicitly below for the selected model, so the generic
    # AST walker must not expand every factory branch here.
    "custom_models/src/benchmark_v2/model_factories/registry.py",
    # Graph compatibility exports use the same fixed name -> module map.
    "custom_models/src/benchmark_v2/models/graph_models/__init__.py",
}
_CLOSURE_PATH_CACHE: dict[tuple[str, str], tuple[list[tuple[str, str]], list[str]]] = {}


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


def _module_name_for_path(relative_path: str) -> tuple[str, Path] | None:
    """Return the import name and import root for a repository Python file."""

    normalized = normalize_relative_path(relative_path)
    for prefix, root_name in (
        ("custom_models/src/", "custom_models/src"),
        ("Time-Series-Library/", "Time-Series-Library"),
    ):
        if not normalized.startswith(prefix):
            continue
        suffix = normalized[len(prefix) :]
        if not suffix.endswith(".py"):
            return None
        parts = suffix[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(part for part in parts if part)
        return module_name, Path(root_name)
    return None


def _module_file_candidates(
    module_name: str, *, project_root: Path, preferred_root: Path | None = None
) -> list[tuple[str, Path]]:
    """Resolve a module name against the repository-local Python roots."""

    module_name = module_name.strip(".")
    if not module_name:
        return []
    parts = module_name.split(".")
    roots: list[Path] = []
    if preferred_root is not None:
        roots.append(project_root / preferred_root)
    if parts[0] == "benchmark_v2":
        roots.append(project_root / "custom_models/src")
    if parts[0] in _REPO_LOCAL_IMPORT_ROOTS:
        roots.append(project_root / "Time-Series-Library")
        roots.append(project_root / "custom_models/src")
    roots.append(project_root)
    seen_roots: set[Path] = set()
    result: list[tuple[str, Path]] = []
    for root in roots:
        root = root.resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        candidate = root.joinpath(*parts)
        file_candidate = candidate.with_suffix(".py")
        package_candidate = candidate / "__init__.py"
        if file_candidate.is_file():
            result.append((file_candidate.relative_to(project_root).as_posix(), file_candidate))
        if package_candidate.is_file():
            result.append((package_candidate.relative_to(project_root).as_posix(), package_candidate))
    return result


def _resolve_import_module(
    current_path: str,
    module_name: str | None,
    level: int,
    imported_names: Iterable[str],
    *,
    project_root: Path,
) -> tuple[list[str], list[str]]:
    """Resolve AST imports that point into this repository.

    The resolver intentionally ignores third-party imports.  Relative imports
    and known repository package roots are local by definition and fail closed
    when their target cannot be resolved.
    """

    context = _module_name_for_path(current_path)
    if context is None:
        return [], []
    current_module, preferred_root = context
    package_parts = (
        current_module.split(".")
        if normalize_relative_path(current_path).endswith("/__init__.py")
        else current_module.split(".")[:-1]
    )
    if level:
        if level > len(package_parts) + 1:
            raise ModelSourceIdentityError(
                f"Relative import escapes the package: {current_path}"
            )
        base_parts = package_parts[: len(package_parts) - (level - 1)]
        if module_name:
            base_parts.extend(module_name.split("."))
        base_module = ".".join(base_parts)
    else:
        base_module = module_name or ""
    local_hint = bool(level) or (
        base_module and base_module.split(".", 1)[0] in _REPO_LOCAL_IMPORT_ROOTS
    )
    resolved: list[str] = []
    base_resolved = False
    if base_module:
        base_candidates = _module_file_candidates(
            base_module, project_root=project_root, preferred_root=preferred_root
        )
        base_resolved = bool(base_candidates)
        resolved.extend(path for path, _ in base_candidates)
    for name in imported_names:
        if not name or name == "*":
            continue
        candidate_module = ".".join(part for part in (base_module, name) if part)
        candidates = _module_file_candidates(
            candidate_module, project_root=project_root, preferred_root=preferred_root
        )
        if candidates:
            resolved.extend(path for path, _ in candidates)
        elif local_hint and level and not base_resolved:
            raise ModelSourceIdentityError(
                f"Unresolved repository-relative import {candidate_module!r} "
                f"from {current_path}"
            )
    if local_hint and not resolved:
        raise ModelSourceIdentityError(
            f"Unresolved repository-local import {base_module!r} from {current_path}"
        )
    return sorted(set(resolved)), []


def _dynamic_import_targets(tree: ast.AST, current_path: str) -> list[str]:
    """Return literal dynamic-import targets; reject non-literal imports."""

    targets: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if function_name not in {"import_module", "__import__"}:
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            if normalize_relative_path(current_path) in _DYNAMIC_IMPORT_ALLOWLIST:
                continue
            raise ModelSourceIdentityError(
                f"Dynamic import cannot be resolved safely in {current_path}"
            )
        targets.append(str(node.args[0].value))
    return targets


def _transitive_closure_paths(
    seeds: Iterable[tuple[str, str]], *, project_root: Path = PROJECT_ROOT
) -> tuple[list[tuple[str, str]], list[str]]:
    """Build a stable repository-local AST import closure from explicit seeds."""

    ordered: dict[str, str] = {}
    visiting: list[str] = []
    visited: set[str] = set()
    cycles: set[str] = set()

    def add_seed(path: str, role: str) -> None:
        normalized = normalize_relative_path(path)
        ordered.setdefault(normalized, role)

    def visit(relative_path: str) -> None:
        normalized = normalize_relative_path(relative_path)
        if normalized in visiting:
            cycle = " -> ".join((*visiting[visiting.index(normalized) :], normalized))
            cycles.add(cycle)
            return
        if normalized in visited:
            return
        resolved = _resolve_repo_file(normalized, project_root)
        visited.add(normalized)
        visiting.append(normalized)
        try:
            try:
                tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=normalized)
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise ModelSourceIdentityError(
                    f"Cannot parse source closure file: {normalized}"
                ) from exc
            imports: list[tuple[str | None, int, tuple[str, ...]]] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, 0, ()))
                elif isinstance(node, ast.ImportFrom):
                    imports.append(
                        (
                            node.module,
                            int(node.level),
                            tuple(alias.name for alias in node.names),
                        )
                    )
            for target in _dynamic_import_targets(tree, normalized):
                imports.append((target, 0, ()))
            children: list[str] = []
            for module_name, level, names in imports:
                paths, _ = _resolve_import_module(
                    normalized,
                    module_name,
                    level,
                    names,
                    project_root=project_root,
                )
                children.extend(paths)
            for child in sorted(set(children)):
                ordered.setdefault(child, "transitive repository-local import")
                visit(child)
        finally:
            visiting.pop()

    for path, role in seeds:
        add_seed(path, role)
    for path in sorted(ordered):
        visit(path)
    return sorted(ordered.items()), sorted(cycles)


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
    factory_module = factory_module_path(model_id)
    add(
        _module_path(factory_module),
        "selected fixed model factory boundary",
    )
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
    cache_key = (registry_entry.canonical_id, str(PROJECT_ROOT))
    cached = _CLOSURE_PATH_CACHE.get(cache_key)
    if cached is None:
        cached = _transitive_closure_paths(
            _closure_paths(registry_entry.canonical_id, registry_entry),
            project_root=PROJECT_ROOT,
        )
        _CLOSURE_PATH_CACHE[cache_key] = cached
    paths, cycles = cached
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
        "source_closure_cycles": cycles,
    }


def canonical_model_source_hash(model_id: str) -> str:
    return str(canonical_model_source_identity(model_id)["canonical_combined_hash"])


def canonical_model_source_closure_manifest(model_id: str) -> dict[str, Any]:
    return canonical_model_source_identity(model_id)
