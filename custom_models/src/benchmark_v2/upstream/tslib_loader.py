from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ..errors import ContractError


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TSLIB_ROOT = (PROJECT_ROOT / "Time-Series-Library").resolve()
TSLIB_LICENSE = (TSLIB_ROOT / "LICENSE").resolve()
_SOURCE_NAMES = {
    "dlinear": "DLinear.py",
    "lightts": "LightTS.py",
    "tide": "TiDE.py",
    "segrnn": "SegRNN.py",
    "transformer": "Transformer.py",
    "patchtst": "PatchTST.py",
    "itransformer": "iTransformer.py",
    "timexer": "TimeXer.py",
    "timesnet": "TimesNet.py",
    "micn": "MICN.py",
    "wpmixer": "WPMixer.py",
    "multipatchformer": "MultiPatchFormer.py",
    "timemixer": "TimeMixer.py",
    "tsmixer": "TSMixer.py",
    "frets": "FreTS.py",
    "crossformer": "Crossformer.py",
    "msgnet": "MSGNet.py",
    "timefilter": "TimeFilter.py",
}
ALLOWED_TSLIB_MODELS = tuple(_SOURCE_NAMES)
_LOAD_LOCK = threading.RLock()
_MODULE_CACHE: dict[str, ModuleType] = {}


@dataclass(frozen=True)
class TSLibSource:
    model_id: str
    upstream_project: str
    source_path: str
    source_relative_path: str
    source_sha256: str
    license_path: str
    license_sha256: str
    source_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_id(model_id: str) -> str:
    key = str(model_id).casefold()
    if key not in _SOURCE_NAMES:
        raise ContractError(
            f"TSLib model {model_id!r} is not allowlisted; "
            f"allowed={list(ALLOWED_TSLIB_MODELS)}"
        )
    return key


def resolve_tslib_source(model_id: str) -> TSLibSource:
    key = _allowed_id(model_id)
    if not TSLIB_ROOT.is_dir() or not TSLIB_LICENSE.is_file():
        raise ContractError(
            f"Local audited Time-Series-Library is incomplete: {TSLIB_ROOT}"
        )
    source = (TSLIB_ROOT / "models" / _SOURCE_NAMES[key]).resolve()
    expected_parent = (TSLIB_ROOT / "models").resolve()
    if source.parent != expected_parent or not source.is_file():
        raise ContractError(f"Audited TSLib source path is invalid: {source}")
    return TSLibSource(
        model_id=key,
        upstream_project="THUML/Time-Series-Library",
        source_path=str(source),
        source_relative_path=source.relative_to(PROJECT_ROOT).as_posix(),
        source_sha256=_sha256(source),
        license_path=str(TSLIB_LICENSE),
        license_sha256=_sha256(TSLIB_LICENSE),
        source_modified=False,
    )


def _reject_foreign_layers_package() -> None:
    root_text = str(TSLIB_ROOT).casefold()
    for name, module in tuple(sys.modules.items()):
        if name != "layers" and not name.startswith("layers."):
            continue
        location = getattr(module, "__file__", None)
        if location and root_text not in str(Path(location).resolve()).casefold():
            raise ContractError(
                f"Refusing TSLib import because foreign module {name!r} "
                f"is already loaded from {location}"
            )


def load_tslib_model_class(model_id: str):
    key = _allowed_id(model_id)
    with _LOAD_LOCK:
        source = resolve_tslib_source(key)
        module = _MODULE_CACHE.get(key)
        if module is None:
            _reject_foreign_layers_package()
            module_name = (
                f"_benchmark_v2_tslib_{key}_{source.source_sha256[:12]}"
            )
            spec = importlib.util.spec_from_file_location(
                module_name, source.source_path
            )
            if spec is None or spec.loader is None:
                raise ContractError(
                    f"Cannot create a local import spec for {source.source_path}"
                )
            module = importlib.util.module_from_spec(spec)
            original_sys_path = list(sys.path)
            sys.modules[module_name] = module
            try:
                sys.path.insert(0, str(TSLIB_ROOT))
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            finally:
                sys.path[:] = original_sys_path
            actual_path = Path(getattr(module, "__file__", "")).resolve()
            if actual_path != Path(source.source_path):
                sys.modules.pop(module_name, None)
                raise ContractError(
                    f"TSLib loader resolved unexpected source {actual_path}"
                )
            _MODULE_CACHE[key] = module
        model_class = getattr(module, "Model", None)
        if model_class is None:
            raise ContractError(
                f"Audited TSLib source has no Model class: {source.source_path}"
            )
        return model_class, source
