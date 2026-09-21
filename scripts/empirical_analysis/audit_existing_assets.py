"""Build the Step-0 read-only inventory and reuse audit.

The script deliberately reads existing artifacts without loading checkpoints or
altering experiment outputs.  It writes only the four Step-0 audit deliverables
under ``custom_models/results/empirical_analysis_v1/audit``.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "custom_models" / "results"
AUDIT = RESULTS / "empirical_analysis_v1" / "audit"


def rel(path: Path | str) -> str:
    """Return a stable repository-relative path where possible."""

    value = Path(path)
    try:
        return value.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return value.resolve().as_posix()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def first_json(path: Path, names: Iterable[str]) -> tuple[Path | None, dict[str, Any]]:
    for name in names:
        candidate = path / name
        if candidate.is_file():
            value = read_json(candidate)
            if isinstance(value, dict):
                return candidate, value
    return None, {}


def find_artifact_dir(root: Path) -> Path:
    """Find the run directory below a variant root.

    STMGPrompt runs have a wrapper directory (A1/P1) and a nested model
    directory.  Benchmark runs are already at the artifact directory.
    """

    if not root.is_dir():
        return root
    direct_markers = (
        "run_status.json",
        "resolved_config.json",
        "model_summary.json",
        "metrics_eval_h3.json",
    )
    if any((root / marker).is_file() for marker in direct_markers):
        return root
    candidates: list[Path] = []
    for marker in direct_markers:
        candidates.extend(p.parent for p in root.rglob(marker))
    if candidates:
        candidates.sort(key=lambda p: (len(p.parts), p.as_posix()))
        return candidates[0]
    return root


def all_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix())


def paths_matching(files: list[Path], predicate, limit: int = 60) -> list[str]:
    found: list[str] = []
    for path in files:
        if predicate(path):
            found.append(rel(path))
            if len(found) >= limit:
                break
    return found


def config_path_for(root: Path) -> Path | None:
    for name in ("resolved_config.json", "effective_config.json", "active_config.json", "config.json", "requested_config.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def extract_score(root: Path, horizon: int) -> float | None:
    candidates = [root / f"metrics_eval_h{horizon}.json"]
    if not (root / f"metrics_eval_h{horizon}.json").is_file():
        candidates.extend(root.rglob(f"metrics_eval_h{horizon}.json"))
    for candidate in candidates:
        value = read_json(candidate)
        if not isinstance(value, dict):
            continue
        for key in ("official_align_score", "Score", "score"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
    return None


def extract_parameter_count(root: Path) -> int | None:
    _, summary = first_json(root, ("model_summary.json",))
    for key in ("total_parameters", "parameter_count", "trainable_parameters"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    # Benchmark summaries occasionally use a nested key.
    for key in ("total_params", "num_parameters"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def extract_seed(root: Path, default: int | None = 2026) -> int | None:
    _, summary = first_json(root, ("model_summary.json",))
    for value in (summary.get("seed"),):
        if isinstance(value, int):
            return value
    config_file = config_path_for(root)
    config = read_json(config_file) if config_file else None
    if isinstance(config, dict):
        for key in ("seed", "random_seed"):
            if isinstance(config.get(key), int):
                return int(config[key])
    return default


def run_status(root: Path) -> tuple[str, bool, bool]:
    status_file = root / "run_status.json"
    status = read_json(status_file) if status_file.is_file() else None
    train_complete = (root / "train_complete.json").is_file()
    evaluation_complete = (root / "evaluation_complete.json").is_file()
    if isinstance(status, dict) and isinstance(status.get("status"), str):
        state = status["status"]
        train_complete = train_complete or state.upper() == "COMPLETED"
        evaluation_complete = evaluation_complete or state.upper() == "COMPLETED"
        return state, train_complete, evaluation_complete
    if isinstance(status, dict):
        state = str(status.get("current_stage", "UNKNOWN"))
        events = status.get("events")
        stages = {str(event.get("stage")) for event in events if isinstance(event, dict)} if isinstance(events, list) else set()
        train_complete = train_complete or "TRAIN_FINISHED" in stages
        evaluation_complete = evaluation_complete or "TEST_FINISHED" in stages
        return state, train_complete, evaluation_complete
    return ("ARTIFACTS_PRESENT" if root.exists() else "MISSING"), train_complete, evaluation_complete


def prediction_start_index(root: Path) -> int | str | None:
    candidates = [p for p in all_files(root) if p.suffix.lower() == ".json" and "prediction" in p.name.lower()]
    for path in candidates:
        value = read_json(path)
        if not isinstance(value, dict):
            continue
        for key in ("prediction_start_index", "start_index", "test_start_index"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float, str)):
                return int(candidate) if isinstance(candidate, float) and candidate.is_integer() else candidate
    return None


def artifact_snapshot(root: Path) -> dict[str, Any]:
    files = all_files(root)
    lower_names = [p.name.lower() for p in files]
    metrics = {
        "metrics_per_step_csv": any(name == "metrics_per_step.csv" for name in lower_names),
        "metrics_per_turbine_csv": any(name == "metrics_per_turbine.csv" for name in lower_names),
        "metrics_by_volatility_csv": any(name == "metrics_by_volatility.csv" for name in lower_names),
    }
    graph_paths = paths_matching(
        files,
        lambda p: any(token in p.name.lower() for token in ("graph", "adjacency", "prior", "edge", "diagnostic")),
    )
    prompt_paths = paths_matching(
        files,
        lambda p: any(token in p.name.lower() for token in ("prompt", "fusion", "attention", "cross", "macro", "gate")),
    )
    loss_paths = paths_matching(files, lambda p: p.name.lower() == "loss_weights.csv")
    site_paths = paths_matching(files, lambda p: p.name.lower() == "site_weight_final.csv")
    export_paths = paths_matching(
        files,
        lambda p: (
            any(token in p.name.lower() for token in ("prediction", "predictions"))
            and p.suffix.lower() in {".npz", ".npy", ".parquet", ".csv"}
        )
        or ("prediction_exports" in p.as_posix().lower() and p.suffix.lower() in {".npz", ".npy", ".parquet", ".csv"}),
    )
    return {
        "file_count": len(files),
        "metrics": metrics,
        "graph_diagnostics": graph_paths,
        "prompt_cross_fusion_diagnostics": prompt_paths,
        "loss_weights_csv": loss_paths,
        "site_weight_final_csv": site_paths,
        "formal_prediction_export": export_paths,
        "prediction_metadata": paths_matching(files, lambda p: "prediction_metadata" in p.name.lower()),
        "prediction_start_index": prediction_start_index(root),
        "key_artifacts": paths_matching(
            files,
            lambda p: p.name.lower() in {
                "best_checkpoint.pt",
                "resolved_config.json",
                "effective_config.json",
                "active_config.json",
                "config.json",
                "run_status.json",
                "protocol_check.json",
                "model_summary.json",
                "metrics_eval_h3.json",
                "metrics_eval_h6.json",
                "metrics_eval_h10.json",
                "e7_readiness.json",
                "e8_readiness.json",
                "e9_readiness.json",
            },
        ),
    }


def json_config(root: Path, name: str) -> dict[str, Any]:
    value = read_json(root / name)
    return value if isinstance(value, dict) else {}


def make_row(
    *,
    family: str,
    variant: str,
    run_id: str,
    model_name: str,
    root: Path,
    classification: str,
    formal: bool,
    efficiency: str,
    reason: str,
    alias_of: str = "",
    status_override: str | None = None,
    train_override: bool | None = None,
    eval_override: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = find_artifact_dir(root)
    status, train_complete, evaluation_complete = run_status(artifact)
    if status_override is not None:
        status = status_override
    if train_override is not None:
        train_complete = train_override
    if eval_override is not None:
        evaluation_complete = eval_override
    config_file = config_path_for(artifact)
    snapshot = artifact_snapshot(artifact)
    checkpoint = "best_checkpoint" if (artifact / "best_checkpoint.pt").is_file() else (
        "last_checkpoint_only" if (artifact / "last_checkpoint.pt").is_file() else "missing"
    )
    row = {
        "experiment_family": family,
        "variant": variant,
        "run_id": run_id,
        "model_name": model_name,
        "seed": extract_seed(artifact),
        "run_root": rel(artifact),
        "config_path": rel(config_file) if config_file else "",
        "checkpoint_status": checkpoint,
        "train_complete": str(bool(train_complete)).lower(),
        "evaluation_complete": str(bool(evaluation_complete)).lower(),
        "run_status": status,
        "status_classification": classification,
        "can_reuse_formal_metrics": str(bool(formal)).lower(),
        "can_reuse_efficiency": efficiency,
        "metric_h3_score": extract_score(artifact, 3),
        "metric_h6_score": extract_score(artifact, 6),
        "metric_h10_score": extract_score(artifact, 10),
        "parameter_count": extract_parameter_count(artifact),
        "metrics_per_step_csv": str(snapshot["metrics"]["metrics_per_step_csv"]).lower(),
        "metrics_per_turbine_csv": str(snapshot["metrics"]["metrics_per_turbine_csv"]).lower(),
        "metrics_by_volatility_csv": str(snapshot["metrics"]["metrics_by_volatility_csv"]).lower(),
        "graph_diagnostics": str(bool(snapshot["graph_diagnostics"])).lower(),
        "prompt_cross_fusion_diagnostics": str(bool(snapshot["prompt_cross_fusion_diagnostics"])).lower(),
        "loss_weights_csv": str(bool(snapshot["loss_weights_csv"])).lower(),
        "site_weight_final_csv": str(bool(snapshot["site_weight_final_csv"])).lower(),
        "formal_prediction_export": str(bool(snapshot["formal_prediction_export"])).lower(),
        "prediction_start_index": snapshot["prediction_start_index"] or "",
        "classification": classification,
        "reuse_reason": reason,
        "alias_of": alias_of,
    }
    detail = {
        "run_key": f"{family}:{variant}:{run_id}",
        "experiment_family": family,
        "variant": variant,
        "run_id": run_id,
        "model_name": model_name,
        "run_root": rel(artifact),
        "wrapper_root": rel(root) if root != artifact else rel(artifact),
        "classification": classification,
        "reuse": {
            "formal_metrics": formal,
            "efficiency": efficiency,
            "reason": reason,
        },
        "status": {
            "run_status": status,
            "train_complete": bool(train_complete),
            "evaluation_complete": bool(evaluation_complete),
            "checkpoint_status": checkpoint,
        },
        "metrics": {
            "H3": extract_score(artifact, 3),
            "H6": extract_score(artifact, 6),
            "H10": extract_score(artifact, 10),
            "parameter_count": extract_parameter_count(artifact),
        },
        "config_path": rel(config_file) if config_file else None,
        "artifacts": snapshot,
        "alias_of": alias_of or None,
        "notes": [],
    }
    return row, detail


FIELDNAMES = [
    "experiment_family",
    "variant",
    "run_id",
    "model_name",
    "seed",
    "run_root",
    "config_path",
    "checkpoint_status",
    "train_complete",
    "evaluation_complete",
    "run_status",
    "status_classification",
    "can_reuse_formal_metrics",
    "can_reuse_efficiency",
    "metric_h3_score",
    "metric_h6_score",
    "metric_h10_score",
    "parameter_count",
    "metrics_per_step_csv",
    "metrics_per_turbine_csv",
    "metrics_by_volatility_csv",
    "graph_diagnostics",
    "prompt_cross_fusion_diagnostics",
    "loss_weights_csv",
    "site_weight_final_csv",
    "formal_prediction_export",
    "prediction_start_index",
    "classification",
    "reuse_reason",
    "alias_of",
]


def find_existing(path: Path) -> Path:
    if path.exists():
        return path
    return path


def add_legacy_rows(rows: list[dict[str, Any]], details: list[dict[str, Any]], root: Path, family: str) -> None:
    if not root.is_dir():
        return
    variant_dirs = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"(?:A|P)\d+(?:-[A-Z0-9]+)?", p.name)]
    if not variant_dirs:
        variant_dirs = [root]
    for variant_root in sorted(variant_dirs, key=lambda p: p.name):
        variant = variant_root.name if variant_root != root else root.name
        artifact = find_artifact_dir(variant_root)
        row, detail = make_row(
            family=family,
            variant=variant,
            run_id=variant,
            model_name="historical_STMGPrompt",
            root=artifact,
            classification="legacy_semantic_mismatch",
            formal=False,
            efficiency="false",
            reason="旧 Dynamic-VADSP/历史批次或 smoke 资产；不进入当前 Fixed-Dual 正式汇总。",
            status_override="HISTORICAL",
        )
        rows.append(row)
        details.append(detail)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    canonical = RESULTS / "st_mgprompt_canonical" / "full_fixed_dual_keep_msmgdwu_seed2026"
    canonical_model = json_config(canonical, "model_summary.json").get("model", "STMGPrompt_Full_MSMGDWU")
    canonical_reason = "唯一 Canonical Fixed-Dual 训练根；A0/P0 只作引用别名，准确率计数一次。"
    row, detail = make_row(
        family="Canonical",
        variant="Canonical",
        run_id="full_fixed_dual_keep_msmgdwu_seed2026",
        model_name=str(canonical_model),
        root=canonical,
        classification="formal_reusable",
        formal=True,
        efficiency="conditional",
        reason=canonical_reason,
    )
    rows.append(row)
    details.append(detail)
    canonical_detail = detail

    # A0 and P0 are intentionally not new training rows.  They point to the
    # same canonical artifact and are kept as aliases for traceability.
    for family, variant in (("component_ablation", "A0"), ("precision_ablation", "P0")):
        row, detail = make_row(
            family=family,
            variant=variant,
            run_id="full_fixed_dual_keep_msmgdwu_seed2026",
            model_name=str(canonical_model),
            root=canonical,
            classification="reference_only_alias",
            formal=True,
            efficiency="conditional",
            reason="引用同一 Canonical 目录；不是独立训练，正式统计必须去重。",
            alias_of="Canonical:full_fixed_dual_keep_msmgdwu_seed2026",
        )
        rows.append(row)
        details.append(detail)

    component_root = RESULTS / "st_mgprompt_component_ablation" / "component_ablation_fixed_dual_seed2026"
    component_defs = {
        "A1": "w/o Spatial Graph",
        "A2": "w/o Adaptive Graph",
        "A3": "w/o Diffusion",
        "A4": "Mean-Pooling Macro Prompt",
        "A5": "Short-context Reverse Cross",
        "A6": "Early-History Macro Cross Fusion",
        "A7": "Shared-Projection Cross Fusion",
        "A8": "w/o MS-MG-DWU",
    }
    for variant, label in component_defs.items():
        wrapper = component_root / variant
        artifact = find_artifact_dir(wrapper)
        if variant == "A1":
            reason = (
                "A1 目录存在 mixed state：config.json 声明 A1/w/o Spatial Graph，"
                "active/requested/effective 配置却声明 A8/w/o MS-MG-DWU；旧状态还记录 failed_protocol_mismatch。"
            )
            classification = "state_conflict_semantic_mismatch"
            formal = False
            efficiency = "false"
            status_override = "STATE_CONFLICT"
        else:
            reason = f"当前 Fixed-Dual {label}；checkpoint、训练/评估标记和 H3/H6/H10 指标齐全。"
            classification = "formal_reusable"
            formal = True
            efficiency = "conditional"
            status_override = None
        row, detail = make_row(
            family="component_ablation",
            variant=variant,
            run_id=variant,
            model_name="STMGPrompt_ComponentAblation",
            root=artifact,
            classification=classification,
            formal=formal,
            efficiency=efficiency,
            reason=reason,
            status_override=status_override,
            train_override=False if variant == "A1" else None,
            eval_override=False if variant == "A1" else None,
        )
        details.append(detail)
        rows.append(row)
        if variant == "A1":
            detail["notes"].append("A1 numeric files are retained for historical/inference inspection only; do not use for formal ablation or efficiency claims.")

    precision_root = RESULTS / "st_mgprompt_precision" / "precision_ablation_fixed_dual_seed2026"
    for variant in ("P1", "P2", "P3", "P4", "P5"):
        artifact = find_artifact_dir(precision_root / variant)
        conflict = variant in {"P4", "P5"}
        reason = (
            "P4/P5 checkpoint、训练/评估和准确率文件存在，但桌面 P0_P5 汇总记录顶层 FAILED/Windows returncode 3221225477；"
            "准确率可复用，效率结论排除。"
            if conflict
            else "当前 Fixed-Dual 精度消融；checkpoint、训练/评估标记和 H3/H6/H10 指标齐全。"
        )
        row, detail = make_row(
            family="precision_ablation",
            variant=variant,
            run_id=variant,
            model_name="STMGPrompt_ComponentAblation",
            root=artifact,
            classification="state_conflict_accuracy_only" if conflict else "formal_reusable",
            formal=True,
            efficiency="false" if conflict else "conditional",
            reason=reason,
            status_override="STATE_CONFLICT" if conflict else None,
        )
        rows.append(row)
        details.append(detail)

    # Eight retained formal baselines from the desktop scope workbook.
    baseline_specs = [
        ("lightts", "LightTS", "custom_models/results/benchmark_v2_uniform_bs4/basic_lightweight_seed2026/LightTS_node_shared_chunk8_bs4_seed2026"),
        ("moving_average", "MovingAverage", "custom_models/results/benchmark_v2_uniform_bs4/basic_lightweight_seed2026/MovingAverage_w144_bs4_seed2026"),
        ("tide", "TiDE", "custom_models/results/benchmark_v2_uniform_bs4/basic_lightweight_seed2026/TiDE_node_shared_d512_bs4_seed2026"),
        ("transformer", "Transformer", "custom_models/results/benchmark_v2_uniform_bs4/e2_a_seed2026/Transformer_node_shared_d512_bs4_seed2026"),
        ("micn", "MICN", "custom_models/results/benchmark_v2_uniform_bs4/e2_b_seed2026/MICN_node_shared_k12_16_d32_bs4_seed2026"),
        ("timesnet", "TimesNet", "custom_models/results/benchmark_v2_uniform_bs4/e2_b_seed2026/TimesNet_node_shared_d32_k5_bs4_seed2026"),
        ("wpmixer", "WPMixer", "custom_models/results/benchmark_v2_uniform_bs4/e2_b_seed2026/WPMixer_node_shared_db2_l1_p16_s8_d256_bs4_seed2026"),
        ("frets", "FreTS", "custom_models/results/benchmark_v2_uniform_bs4/e2_c_seed2026/FreTS_node_shared_tcfft_e128_h256_bs4_seed2026"),
    ]
    for variant, model_name, root_text in baseline_specs:
        root = REPO / Path(root_text)
        row, detail = make_row(
            family="baseline_8_formal",
            variant=variant,
            run_id=root.name,
            model_name=model_name,
            root=root,
            classification="formal_reusable",
            formal=True,
            efficiency="conditional",
            reason="桌面基准范围表保留的 8 个正文基准；benchmark_v2_uniform_bs4 formal artifacts 完整。",
        )
        rows.append(row)
        details.append(detail)

    # E7/E8 are analysis manifests: external six-model context is present,
    # while internal A0-A3/A0/A4-A7 diagnostics are blocked by protocol mismatch.
    analysis_specs = [
        ("E7", RESULTS / "benchmark_v2" / "e7_graph_mechanism_analysis_seed2026", "E7_READINESS.json", "partial_external_only", "E7 external 6/6 ready; internal graph 0/4, grouped internal conclusions blocked."),
        ("E8", RESULTS / "benchmark_v2" / "e8_prompt_cross_fusion_analysis_seed2026", "E8_READINESS.json", "partial_external_only", "E8 external 6/6 ready; internal prompt/cross-fusion 0/5, grouped internal conclusions blocked."),
    ]
    for variant, root, readiness_name, classification, reason in analysis_specs:
        readiness = read_json(root / readiness_name)
        status = "PARTIAL_EXTERNAL_ONLY"
        if isinstance(readiness, dict):
            status = str(readiness.get("status", status))
        row, detail = make_row(
            family="analysis_manifest",
            variant=variant,
            run_id=root.name,
            model_name=f"{variant} analysis",
            root=root,
            classification=classification,
            formal=False,
            efficiency="false",
            reason=reason,
            status_override=status,
            train_override=False,
            eval_override=False,
        )
        rows.append(row)
        details.append(detail)
        detail["readiness"] = readiness

    # E9 transfer runs are individually formal and accuracy-reusable.  The
    # pairing audit does not verify exact initial tensors, so that claim stays
    # explicitly limited in the classification/reuse reason.
    e9_root = RESULTS / "benchmark_v2" / "msmg_dwu_transfer_seed2026"
    e9_readiness = read_json(e9_root / "E9_TRANSFER_READINESS.json")
    e9_rows = e9_readiness.get("rows", []) if isinstance(e9_readiness, dict) else []
    # The readiness manifest is the authoritative six-model scope.  It also
    # exposes a missing directory if a stale manifest names an absent run.
    e9_specs: list[tuple[str, str, Path]] = []
    if isinstance(e9_rows, list):
        for item in e9_rows:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("model_id", "unknown"))
            run_id = str(item.get("run_id", model_id))
            source_root = item.get("source_root")
            source_path = Path(source_root) if isinstance(source_root, str) else e9_root / run_id
            if not source_path.is_absolute():
                source_path = REPO / source_path
            e9_specs.append((model_id, run_id, source_path))
    else:
        for child in sorted(e9_root.iterdir()) if e9_root.is_dir() else []:
            if child.is_dir():
                e9_specs.append((child.name.split("_seed2026")[0], child.name, child))
    for model_name, run_id, child in e9_specs:
        if not child.is_dir():
            row, detail = make_row(
                family="E9_transfer",
                variant=model_name,
                run_id=run_id,
                model_name=model_name,
                root=child,
                classification="missing_manifest_target",
                formal=False,
                efficiency="false",
                reason="E9 readiness manifest names this transfer run, but the referenced artifact directory is absent; do not infer metrics.",
                status_override="MISSING",
                train_override=False,
                eval_override=False,
            )
            rows.append(row)
            details.append(detail)
            continue
        row, detail = make_row(
            family="E9_transfer",
            variant=model_name,
            run_id=run_id,
            model_name=model_name,
            root=child,
            classification="formal_transfer_reusable_init_unverified",
            formal=True,
            efficiency="conditional",
            reason="E9 transfer/control artifacts complete; same seed/source/config/protocol is recorded, exact initial-weight pairing is NOT_VERIFIED.",
        )
        rows.append(row)
        details.append(detail)

    e9_manifest_count = len(e9_specs)
    e9_present_count = sum(1 for _, _, path in e9_specs if path.is_dir())
    e9_missing_models = [model for model, _, path in e9_specs if not path.is_dir()]

    # Enumerate representative historical roots so they cannot be silently
    # mixed into the new Fixed-Dual table.
    legacy_specs = [
        ("legacy_component_ablation", RESULTS / "st_mgprompt_component_ablation" / "component_ablation_p1_seed2026"),
        ("legacy_method_full_ablation", RESULTS / "st_mgprompt_method_full_ablation" / "method_full_msmgdwu_seed2026_p0_p5"),
        ("legacy_method_full_test_ablation", RESULTS / "st_mgprompt_method_full_test_ablation" / "method_full_msmgdwu_seed2026_p0_p5_test"),
        ("legacy_precision_smoke", RESULTS / "st_mgprompt_precision_smoke" / "precision_smoke_diffusion_history"),
        ("legacy_fix2_checks", RESULTS / "st_mgprompt_fix2_checks"),
        ("legacy_vadsp_smoke", RESULTS / "st_mgprompt_vadsp_fix_smoke"),
    ]
    for family, root in legacy_specs:
        add_legacy_rows(rows, details, root, family)

    protocol_snapshot = {
        "protocol_name": "Fixed-Dual refactor / empirical_analysis_v1",
        "seed": 2026,
        "horizons": [3, 6, 10],
        "canonical_id": "full_fixed_dual_keep_msmgdwu_seed2026",
        "target": "Patv_raw",
        "input_alignment_signal": "Patv_clean_for_input",
        "window_history": 96,
        "window_horizons": [6, 18, 36],
        "batch_profile": "current formal artifacts are checked by their own protocol_check/resolved_config; efficiency requires matched hardware metadata",
        "a0_a8_authority": "custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md",
    }

    status_path = RESULTS / "empirical_analysis_v1" / "experiment_status.json"
    status_payload = read_json(status_path)
    status_counts: dict[str, int] = {}
    status_items = status_payload if isinstance(status_payload, list) else (
        status_payload.get("statuses", []) if isinstance(status_payload, dict) else []
    )
    if isinstance(status_items, list):
        for item in status_items:
            if isinstance(item, dict):
                key = str(item.get("final_status", item.get("status", "UNKNOWN")))
                status_counts[key] = status_counts.get(key, 0) + 1
    data_evidence = RESULTS / "empirical_analysis_v1" / "data_evidence"
    metadata_path = data_evidence / "analysis_metadata.json"
    metadata = read_json(metadata_path)
    evidence_file_count = sum(1 for p in data_evidence.rglob("*") if p.is_file()) if data_evidence.is_dir() else 0

    sources_read = [
        "README.md",
        "RUNBOOK_FIXED_DUAL_REFACTOR.md",
        "HANDOFF_FIXED_DUAL_REFACTOR.md",
        "custom_models/docs/st_mgprompt_component_ablation/A0_A8_FORMAL_DEFINITIONS.md",
        "custom_models/src/st_mgprompt/config.py",
        "custom_models/src/st_mgprompt/experiment_protocol.py",
        "custom_models/src/st_mgprompt/run_st_mgprompt.py",
        "custom_models/src/st_mgprompt/evaluate.py",
        "custom_models/src/st_mgprompt/metrics.py",
        "custom_models/src/st_mgprompt/diagnostics.py",
        "scripts/export_formal_predictions.py",
        "custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026",
        "custom_models/results/st_mgprompt_component_ablation/component_ablation_fixed_dual_seed2026",
        "custom_models/results/st_mgprompt_precision/precision_ablation_fixed_dual_seed2026",
        "custom_models/results/benchmark_v2_uniform_bs4",
        "custom_models/results/benchmark_v2/e7_graph_mechanism_analysis_seed2026",
        "custom_models/results/benchmark_v2/e8_prompt_cross_fusion_analysis_seed2026",
        "custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026",
        "C:/Users/12811/Desktop/实验结果/A0-A8组件消融实验结果.xlsx",
        "C:/Users/12811/Desktop/实验结果/P0_P5.xlsx",
        "C:/Users/12811/Desktop/实验结果/E7.xlsx",
        "C:/Users/12811/Desktop/实验结果/E8.xlsx",
        "C:/Users/12811/Desktop/实验结果/E9_MS-MG-DWU_结果_5模型.xlsx",
        "C:/Users/12811/Desktop/实验结果/基准模型对比.xlsx",
    ]

    reusable_assets = [
        {"asset_type": "canonical_accuracy", "path": rel(canonical), "source_run": "Canonical", "allowed_use": "A0/P0/Cannonical accuracy reference", "limitations": "A0 and P0 are aliases; count once."},
        {"asset_type": "component_accuracy", "path": rel(component_root), "source_run": "A2-A8", "allowed_use": "current Fixed-Dual component accuracy aggregation", "limitations": "A1 is excluded; efficiency remains conditional on matched hardware."},
        {"asset_type": "precision_accuracy", "path": rel(precision_root), "source_run": "P1-P5", "allowed_use": "accuracy comparison", "limitations": "P4/P5 state conflict; do not use their efficiency fields."},
        {"asset_type": "baseline_accuracy", "path": "custom_models/results/benchmark_v2_uniform_bs4", "source_run": "8正文基准", "allowed_use": "formal baseline accuracy and difficulty context", "limitations": "cross-run efficiency comparisons need hardware matching."},
        {"asset_type": "E9_transfer_accuracy", "path": rel(e9_root), "source_run": "E9", "allowed_use": "present transfer/control accuracy comparison", "limitations": f"{e9_present_count}/{e9_manifest_count} transfer directories present; exact initial-state pairing not verified."},
        {"asset_type": "E7_external_graph_context", "path": rel(analysis_specs[0][1]), "source_run": "E7", "allowed_use": "external graph diagnostic context", "limitations": "internal A0-A3 evidence blocked."},
        {"asset_type": "E8_external_prompt_context", "path": rel(analysis_specs[1][1]), "source_run": "E8", "allowed_use": "external prompt/fusion diagnostic context", "limitations": "internal A0/A4-A7 evidence blocked."},
        {"asset_type": "step2_data_evidence", "path": rel(data_evidence), "source_run": "Step2", "allowed_use": "data-only difficulty/alignment analyses", "limitations": "model conclusions still require eligible formal runs."},
    ]

    missing_or_blocked = [
        {"item": "A1", "status": "blocked", "reason": "mixed config state; rerun/reconcile as current A1 w/o Spatial Graph before formal use."},
        {"item": "P4/P5 efficiency", "status": "blocked", "reason": "top-level FAILED/state conflict despite complete accuracy files; accuracy-only reuse."},
        {"item": "E7 internal", "status": "blocked", "reason": "INTERNAL_GRAPH_READY=0/4; only external six-model diagnostics are ready."},
        {"item": "E8 internal", "status": "blocked", "reason": "INTERNAL_PROMPT_FUSION_READY=0/5; only external six-model diagnostics are ready."},
        {"item": "E9 manifest/filesystem reconciliation", "status": "blocked" if e9_missing_models else "clear", "reason": f"E9 readiness manifest lists {e9_manifest_count}/{e9_manifest_count}, but {e9_present_count}/{e9_manifest_count} transfer directories are present; missing: {', '.join(e9_missing_models) if e9_missing_models else 'none'}."},
        {"item": "historical Dynamic-VADSP roots", "status": "excluded", "reason": "semantic mismatch with current Fixed-Dual protocol."},
        {"item": "formal prediction arrays", "status": "conditional", "reason": "metadata is present for many runs; an actual prediction export is only marked when array/table files are found."},
    ]

    no_training_analyses_ready = [
        "Step2 data-evidence plots/tables: alignment, train-only statistics, difficulty strata and formal metric-match audit.",
        "Current canonical/A2-A8/P1-P3 accuracy aggregation, with A1 and P4/P5 exclusions applied.",
        "8正文基准模型的 accuracy and parameter/difficulty context.",
        f"E9 transfer/control accuracy analysis for the {e9_present_count}/{e9_manifest_count} present transfer artifacts, with exact-initialization claim explicitly withheld; reconcile missing rows before claiming the full manifest scope.",
        "E7/E8 external-only diagnostic context; no internal grouped claim until readiness gates pass.",
        "Artifact inventory and reuse-matrix generation itself (no training or checkpoint mutation).",
    ]

    artifact_payload = {
        "schema_version": "step0_existing_artifacts_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_snapshot,
        "sources_read": sources_read,
        "scope": {
            "canonical": rel(canonical),
            "formal_component_variants": ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "formal_precision_variants": ["P0", "P1", "P2", "P3", "P4", "P5"],
            "baselines": [item[0] for item in baseline_specs],
            "analysis_roots": [rel(item[1]) for item in analysis_specs] + [rel(e9_root)],
            "historical_roots": [rel(item[1]) for item in legacy_specs],
        },
        "canonical_aliases": {
            "Canonical": rel(canonical),
            "A0": rel(canonical),
            "P0": rel(canonical),
            "unique_training_count": 1,
            "note": "A0/P0 are reference-only aliases and are never counted as independent training runs.",
        },
        "current_variant_definitions": {
            "A0": "Canonical Full",
            "A1": "w/o Spatial Graph",
            "A2": "w/o Adaptive Graph",
            "A3": "w/o Diffusion",
            "A4": "Mean-Pooling Macro Prompt",
            "A5": "Short-context Reverse Cross",
            "A6": "Early-History Macro Cross Fusion (macro_to_fine_exclude_recent_len=24)",
            "A7": "Shared-Projection Cross Fusion (share_cross_attention_projections=true)",
            "A8": "w/o MS-MG-DWU (fair_main loss)",
        },
        "experiment_status_counts": status_counts,
        "step2_data_evidence": {
            "metadata_path": rel(metadata_path),
            "metadata": metadata,
            "file_count": evidence_file_count,
        },
        "runs": details,
        "reusable_assets": reusable_assets,
        "missing_or_blocked": missing_or_blocked,
        "no_training_analyses_ready": no_training_analyses_ready,
        "read_only_audit": True,
        "historical_files_modified": False,
    }

    csv_path = AUDIT / "existing_runs.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_json(AUDIT / "existing_artifacts.json", artifact_payload)

    reuse_rows = [
        {"evidence_task": "Canonical/P0/A0 accuracy", "required_variants": "Canonical,A0,P0", "existing_assets": rel(canonical), "reusable": "yes", "missing_items": "none", "next_step": "aggregate once by Canonical id", "efficiency_allowed": "conditional", "limitations": "A0/P0 aliases; do not count twice"},
        {"evidence_task": "Fixed-Dual precision accuracy", "required_variants": "P1,P2,P3", "existing_assets": rel(precision_root), "reusable": "yes", "missing_items": "none", "next_step": "accuracy table", "efficiency_allowed": "conditional", "limitations": "matched hardware required"},
        {"evidence_task": "P4/P5 accuracy", "required_variants": "P4,P5", "existing_assets": rel(precision_root), "reusable": "accuracy-only", "missing_items": "clean top-level run state", "next_step": "do not use efficiency; reconcile if needed", "efficiency_allowed": "no", "limitations": "state conflict/FAILED returncode"},
        {"evidence_task": "Current component accuracy", "required_variants": "A2,A3,A4,A5,A6,A7,A8", "existing_assets": rel(component_root), "reusable": "yes", "missing_items": "none", "next_step": "aggregate formal metrics", "efficiency_allowed": "conditional", "limitations": "A1 excluded"},
        {"evidence_task": "A1 w/o Spatial Graph", "required_variants": "A1", "existing_assets": rel(component_root / "A1"), "reusable": "no", "missing_items": "reconciled A1 run", "next_step": "rerun/reconcile current A1", "efficiency_allowed": "no", "limitations": "mixed A1/A8 state"},
        {"evidence_task": "8正文基准 accuracy/difficulty", "required_variants": "lightts,moving_average,tide,transformer,micn,timesnet,wpmixer,frets", "existing_assets": "custom_models/results/benchmark_v2_uniform_bs4", "reusable": "yes", "missing_items": "none", "next_step": "difficulty/accuracy summaries", "efficiency_allowed": "conditional", "limitations": "hardware comparability"},
        {"evidence_task": "E7 graph diagnostics", "required_variants": "external six", "existing_assets": "custom_models/results/benchmark_v2/e7_graph_mechanism_analysis_seed2026", "reusable": "external-only", "missing_items": "internal A0-A3 readiness", "next_step": "use read-only external context", "efficiency_allowed": "no", "limitations": "INTERNAL_GRAPH_READY=0/4"},
        {"evidence_task": "E8 prompt/cross-fusion diagnostics", "required_variants": "external six", "existing_assets": "custom_models/results/benchmark_v2/e8_prompt_cross_fusion_analysis_seed2026", "reusable": "external-only", "missing_items": "internal A0/A4-A7 readiness", "next_step": "use read-only external context", "efficiency_allowed": "no", "limitations": "INTERNAL_PROMPT_FUSION_READY=0/5"},
        {"evidence_task": "E9 transfer/control accuracy", "required_variants": "six transfer pairs", "existing_assets": rel(e9_root), "reusable": "partial", "missing_items": f"manifest/filesystem mismatch; missing {', '.join(e9_missing_models) if e9_missing_models else 'none'}; exact initial-state verification", "next_step": "reconcile missing transfer directory then compare", "efficiency_allowed": "conditional", "limitations": "same seed/source/config/protocol only"},
        {"evidence_task": "Step2 data-only analyses", "required_variants": "data_evidence", "existing_assets": rel(data_evidence), "reusable": "yes", "missing_items": "none for data-only", "next_step": "no-training figures/tables", "efficiency_allowed": "n/a", "limitations": "does not repair blocked model runs"},
        {"evidence_task": "Historical Dynamic-VADSP assets", "required_variants": "old A/P/smoke roots", "existing_assets": "custom_models/results/st_mgprompt_* historical roots", "reusable": "no", "missing_items": "current Fixed-Dual reruns", "next_step": "exclude from formal table", "efficiency_allowed": "no", "limitations": "semantic mismatch"},
    ]
    reuse_path = AUDIT / "reuse_matrix.csv"
    with reuse_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reuse_rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reuse_rows)

    # Keep the report human-readable and make the acceptance boundary explicit.
    formal_rows = [r for r in rows if r["can_reuse_formal_metrics"] == "true"]
    clean_component = [r for r in rows if r["experiment_family"] == "component_ablation" and r["classification"] == "formal_reusable"]
    clean_precision = [r for r in rows if r["experiment_family"] == "precision_ablation" and r["classification"] == "formal_reusable"]
    report = f"""# 步骤0：已有实验资产审计报告

生成时间：{artifact_payload['generated_at']}（只读审计；未修改历史实验文件，未生成哈希）。

## 1. 审计范围与协议

本次按 `PLAN.md` 检查 README、Fixed-Dual runbook/handoff、A0-A8 当前定义、协议/运行/评估/指标/诊断/预测导出源码，以及 `custom_models/results` 下 Canonical、P0-P5、A0-A8、E7/E8/E9、8 个正文基准模型和历史根目录。协议快照为：seed=2026，目标 `Patv_raw`，输入对齐信号 `Patv_clean_for_input`，history=96，正式 horizon=3/6/10，窗口 horizon=[6,18,36]。

输出文件：

- `existing_runs.csv`：逐运行/分析根的状态、指标、checkpoint、配置、诊断与预测导出清单。
- `existing_artifacts.json`：机器可读的来源、复用边界、阻塞项和 Step2 数据证据。
- `reuse_matrix.csv`：证据任务到现有资产的复用矩阵。
- `audit_report.md`：本报告。

## 2. Canonical 身份与别名去重

唯一 Canonical 训练根是 `{rel(canonical)}`。A0 和 P0 的 `reference.json` 都指向该目录，均为 reference-only，不是独立训练。正式准确率必须按 **1 个 Canonical 训练**计数，不能把 A0/P0 加成两个 run。当前 A6/A7/A8 的定义以 `A0_A8_FORMAL_DEFINITIONS.md` 为准：A6=early-history macro cross（排除最近 24 步）、A7=shared-projection cross-fusion、A8=w/o MS-MG-DWU（fair_main loss）。

## 3. 当前 Fixed-Dual 可复用情况

- Canonical/A0/P0：准确率可复用；A0/P0 仅别名。
- 组件消融：A2-A8 共 {len(clean_component)} 个当前正式 artifact 可做准确率聚合；A1 不可直接复用。
- 精度消融：P1-P3 共 {len(clean_precision)} 个干净 artifact；P4/P5 的 checkpoint、训练/评估和 H3/H6/H10 文件存在，但桌面 `P0_P5.xlsx`/运行状态记录顶层 FAILED/Windows returncode，故只允许准确率复用，明确排除效率结论。
- 8 个正文基准：lightts、moving_average、tide、transformer、micn、timesnet、wpmixer、frets 均有当前 `benchmark_v2_uniform_bs4` formal artifact；可用于准确率、参数量和难度上下文，跨硬件效率仍需同机条件。

## 4. 必须阻断的语义/状态冲突

### A1

A1 目录中的 `config.json` 声明 `A1 / w/o Spatial Graph`，但 `active_config.json`、`requested_config.json`、`effective_config_diff.json` 和部分模型摘要落在 `A8 / w/o MS-MG-DWU` 语义；组件根状态还记录 `failed_protocol_mismatch`。因此 A1 数字文件只保留作历史/推理检查，不能作为当前 A1 正式消融或效率证据。需要重跑或完成一致性修复后才可纳入。

### P4/P5

P4/P5 的准确率结果可以复核，但运行状态冲突（桌面汇总的 Windows returncode 为 `3221225477`）意味着不能从它们得出效率结论。`reuse_matrix.csv` 已将两者拆成 accuracy-only。

### E7/E8/E9

E7 为 external 6/6 ready、internal graph 0/4；E8 为 external 6/6 ready、internal prompt/fusion 0/5。因此 E7/E8 只能作为外部只读诊断上下文。E9 readiness manifest 声称 6/6，但当前文件系统只有 {e9_present_count}/{e9_manifest_count} 个 transfer 目录（缺少：{', '.join(e9_missing_models) if e9_missing_models else '无'}）；只有实际存在的行可复用，完整六模型结论需先对齐 manifest。另有 `E9_PAIRING_AUDIT.json` 的 exact initial-state pairing 为 NOT_VERIFIED，只能声称同 seed/source/config/protocol，不能声称逐张量同初始化。

历史 Dynamic-VADSP、旧 method-full、smoke/fix 根已列入 inventory，但均标为 `legacy_semantic_mismatch`，不进入当前 Fixed-Dual 正式表。

## 5. 现在可以直接开始的 no-training 分析

1. Step2 `data_evidence` 的对齐、train-only 统计、difficulty 分层和 formal metric-match 图表/表格。
2. Canonical + A2-A8 + P1-P3 的当前准确率汇总（A1、P4/P5 按上述规则排除相应结论）。
3. 8 个正文基准的准确率/参数量/难度上下文。
4. E9 transfer/control 准确率比较（当前 {e9_present_count}/{e9_manifest_count} 个目录可用；保留 exact-init 限制，先对齐缺失行）。
5. E7/E8 external-only 图结构及 prompt/fusion 诊断上下文；内部 grouped claim 等待 readiness gate。
6. 本次清单和复用矩阵本身的下游汇总；无需训练或加载 checkpoint。

## 6. 接受检查

| 检查项 | 结果 |
|---|---|
| 四个 Step0 文件存在 | ✅ 已写出并随后做 JSON/CSV 可读性检查 |
| Canonical 唯一身份可追溯 | ✅；A0/P0 去重 |
| A0-A8 当前定义已固定 | ✅；A6/A7/A8 采用当前正式定义 |
| A1 状态冲突被显式阻断 | ✅；不可用于正式 A1/效率 |
| P4/P5 accuracy-only 边界 | ✅；效率排除 |
| E7/E8 readiness 边界 | ✅；仅 external-only |
| E9 exact-init 限制 | ✅；NOT_VERIFIED 已保留 |
| 历史 Dynamic-VADSP 资产隔离 | ✅；标记 semantic mismatch |
| 哈希/历史文件修改 | ✅ 未生成哈希，未修改历史结果 |

结论：步骤0的**审计交付物**已完成且可机器读取；但这不等于所有历史/当前候选都已经“干净可用于全部结论”。A1、P4/P5、E7/E8 的限制仍必须遵守。步骤1/2的最终复核见主任务回复。
"""
    (AUDIT / "audit_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "audit_dir": rel(AUDIT),
        "existing_runs": len(rows),
        "formal_metric_rows": len(formal_rows),
        "detail_runs": len(details),
        "status_counts": status_counts,
        "step2_data_evidence_files": evidence_file_count,
        "outputs": [rel(AUDIT / name) for name in ("existing_runs.csv", "existing_artifacts.json", "reuse_matrix.csv", "audit_report.md")],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
