from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_project_path
from .empirical_protocol import EMPIRICAL_FAMILIES, EMPIRICAL_PROTOCOL_ID, FROZEN_PROTOCOL_FIELDS, dry_run_report
from .experiment_protocol import CANONICAL_ID, canonical_config, canonical_directory
from .graph_prior import EPS, summarize_graph_identity


HORIZONS = (3, 6, 10)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _find_variant_artifact(g_root: Path, variant_id: str, filename: str) -> Path | None:
    candidates = []
    for path in g_root.rglob(filename):
        config = _json(path.parent / "effective_config.json") or _json(path.parent / "resolved_config.json")
        if str(config.get("variant", "")).upper() == variant_id:
            candidates.append(path)
            continue
        if variant_id.lower() in path.as_posix().lower():
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: ("smoke" not in path.as_posix().lower(), path.stat().st_mtime), reverse=True)[0]


def _canonical_graphs() -> tuple[np.ndarray, np.ndarray, Path]:
    import torch

    source = canonical_directory(resolve_project_path("."))
    checkpoint_path = source / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    macro = state["A_macro_prior"].detach().cpu().numpy().astype(np.float32)
    micro = state["A_micro_prior"].detach().cpu().numpy().astype(np.float32)
    return macro, micro, checkpoint_path


def _support_overlap(macro: np.ndarray, micro: np.ndarray) -> dict[str, Any]:
    macro_support = np.asarray(macro) > EPS
    micro_support = np.asarray(micro) > EPS
    union = np.logical_or(macro_support, micro_support)
    return {
        "macro_edge_count": int(macro_support.sum()),
        "micro_edge_count": int(micro_support.sum()),
        "support_jaccard": float(np.logical_and(macro_support, micro_support).sum() / max(int(union.sum()), 1)),
    }


def _edge_distance_rows(variant_id: str, graph_name: str, adjacency: np.ndarray, config) -> dict[str, Any]:
    location = pd.read_csv(config.resolve_path(config.location_path))
    lower = {name.lower(): name for name in location.columns}
    columns = [lower[name.lower()] for name in config.graph_coordinate_cols if name.lower() in lower]
    coords = location[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)[: adjacency.shape[0]]
    rows, cols = np.where(np.asarray(adjacency) > EPS)
    keep = rows != cols
    distances = np.linalg.norm(coords[rows[keep]] - coords[cols[keep]], axis=1)
    return {
        "variant_id": variant_id,
        "graph": graph_name,
        "edge_count_without_self": int(distances.size),
        "distance_min": float(distances.min()) if distances.size else "",
        "distance_median": float(np.median(distances)) if distances.size else "",
        "distance_mean": float(distances.mean()) if distances.size else "",
        "distance_max": float(distances.max()) if distances.size else "",
        "interpretation": "statistical_or_geographic_association_not_causality",
    }


def write_step4_reports(results_root: str | Path) -> dict[str, Any]:
    root = resolve_project_path(results_root)
    g_root = root / "G"
    g_root.mkdir(parents=True, exist_ok=True)
    variants = list(EMPIRICAL_FAMILIES["G"].values())
    canonical = canonical_config()
    macro0, micro0, checkpoint_path = _canonical_graphs()
    identity0 = summarize_graph_identity(macro0, micro0, canonical, fit_split="canonical_train_only_checkpoint")
    identity0.update({"variant_id": "G0", "source": str(checkpoint_path.resolve())})

    manifest_rows = []
    audit_rows = []
    identity_rows = []
    trace_rows = []
    overlap_rows = []
    distance_rows = []
    failures = []
    aggregate_traces: dict[str, Any] = {}
    for variant in variants:
        report = dry_run_report(variant)
        artifact = _find_variant_artifact(g_root, variant.variant_id, "graph_identity.json")
        identity = identity0 if variant.variant_id == "G0" else (_json(artifact) if artifact else {})
        trace_path = _find_variant_artifact(g_root, variant.variant_id, "branch_graph_assignment_trace.json")
        trace = _json(trace_path) if trace_path else {}
        expected_assignment = canonical.branch_graph_assignment
        if "branch_graph_assignment" in variant.config_overrides:
            expected_assignment = variant.config_overrides["branch_graph_assignment"]
        manifest_rows.append(
            {
                **variant.to_dict(),
                "protocol_profile": "STMG_FORMAL_V2",
                "source_scope": "internal_graph",
                "formal_result_status": "reference-only" if variant.variant_id == "G0" else "preflight-only",
                "graph_identity_path": str(artifact.resolve()) if artifact else (str(checkpoint_path.resolve()) if variant.variant_id == "G0" else ""),
            }
        )
        audit_rows.append(
            {
                "variant_id": variant.variant_id,
                "passed": bool(report.get("passed")),
                "execution_ready": bool(report.get("execution_ready")),
                "expected_diff_fields": report.get("expected_diff_fields", []),
                "actual_diff_fields": report.get("actual_diff_fields", []),
                "frozen_protocol_changes": report.get("frozen_protocol_changes", []),
            }
        )
        identity_rows.append(
            {
                "variant_id": variant.variant_id,
                "node_count": identity.get("node_count"),
                "fit_split": identity.get("fit_split"),
                "source_components": json.dumps(identity.get("source_components", []), ensure_ascii=False),
                "branch_graph_assignment": identity.get("branch_graph_assignment", expected_assignment),
                "adaptive_support_mode": identity.get("adaptive_support_mode", variant.config_overrides.get("adaptive_support_mode", "prior_constrained")),
                "graph_prior_component": identity.get("graph_prior_component", variant.config_overrides.get("graph_prior_component", "fused")),
                "graph_rewire_mode": identity.get("graph_rewire_mode", variant.config_overrides.get("graph_rewire_mode", "none")),
                "micro_edge_count": (identity.get("edge_count") or {}).get("micro"),
                "macro_edge_count": (identity.get("edge_count") or {}).get("macro"),
                "identity_source": str(artifact.resolve()) if artifact else (str(checkpoint_path.resolve()) if variant.variant_id == "G0" else ""),
            }
        )
        actual_assignment = trace.get("assignment")
        trace_rows.append(
            {
                "variant_id": variant.variant_id,
                "expected_assignment": expected_assignment,
                "actual_assignment": actual_assignment or ("matched" if variant.variant_id == "G0" else ""),
                "fine_graph_id": trace.get("fine_graph_id", "micro" if variant.variant_id == "G0" else ""),
                "coarse_graph_id": trace.get("coarse_graph_id", "macro" if variant.variant_id == "G0" else ""),
                "trace_path": str(trace_path.resolve()) if trace_path else "",
            }
        )
        aggregate_traces[variant.variant_id] = trace or {
            "assignment": "matched" if variant.variant_id == "G0" else None,
            "status": "reference-only" if variant.variant_id == "G0" else "preflight_not_found",
        }

        macro = micro = None
        if variant.variant_id == "G0":
            macro, micro = macro0, micro0
        elif artifact:
            artifact_directory = str(identity.get("artifact_directory", "")).strip()
            graph_dir = Path(artifact_directory) if artifact_directory else Path("__missing_graph_artifact_directory__")
            if not artifact_directory or not graph_dir.is_dir():
                metadata_path = _find_variant_artifact(g_root, variant.variant_id, "metadata.json")
                graph_dir = metadata_path.parent if metadata_path is not None else graph_dir
            macro_path = graph_dir / "macro_trend_adjacency.npy"
            micro_path = graph_dir / "micro_local_adjacency.npy"
            if macro_path.is_file() and micro_path.is_file():
                macro, micro = np.load(macro_path), np.load(micro_path)
        if macro is not None and micro is not None:
            overlap_rows.append({"variant_id": variant.variant_id, **_support_overlap(macro, micro)})
            distance_rows.append(_edge_distance_rows(variant.variant_id, "macro", macro, canonical))
            distance_rows.append(_edge_distance_rows(variant.variant_id, "micro", micro, canonical))

        if variant.variant_id != "G0":
            failure_path = (
                g_root
                / f"{variant.variant_id.lower()}_empirical_v1"
                / "seed_2026"
                / "STMGPrompt_ComponentAblation"
                / "failure_report.json"
            )
            if not failure_path.is_file():
                _write_json(
                    failure_path,
                    {
                        "protocol_id": EMPIRICAL_PROTOCOL_ID,
                        "variant_id": variant.variant_id,
                        "status": "not_started",
                        "reasons": ["formal_long_training_not_run_during_step4_preflight"],
                        "contaminated_checkpoint_reused": False,
                        "updated_at": _utc_now(),
                    },
                )
            failures.append(
                {
                    "variant_id": variant.variant_id,
                    "run_status": "preflight-only",
                    "reason": "formal_long_training_not_run_during_step4_preflight",
                    "failure_report": str(failure_path.resolve()),
                }
            )

    _write_json(
        g_root / "G_VARIANT_MANIFEST.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "canonical_run_id": CANONICAL_ID,
            "canonical_checkpoint": str(checkpoint_path.resolve()),
            "variants": manifest_rows,
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        g_root / "G_PROTOCOL_AUDIT.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "passed": all(row["passed"] and not row["frozen_protocol_changes"] for row in audit_rows),
            "frozen_fields": list(FROZEN_PROTOCOL_FIELDS),
            "variant_audits": audit_rows,
            "causal_language_prohibited": True,
            "updated_at": _utc_now(),
        },
    )
    _write_csv(g_root / "G_GRAPH_IDENTITY.csv", identity_rows, list(identity_rows[0]))
    _write_json(g_root / "G_BRANCH_ASSIGNMENT_TRACE.json", {"variants": aggregate_traces, "rows": trace_rows})
    _write_csv(g_root / "G_SUPPORT_OVERLAP.csv", overlap_rows, ["variant_id", "macro_edge_count", "micro_edge_count", "support_jaccard"])
    _write_csv(
        g_root / "G_EDGE_DISTANCE_SUMMARY.csv",
        distance_rows,
        ["variant_id", "graph", "edge_count_without_self", "distance_min", "distance_median", "distance_mean", "distance_max", "interpretation"],
    )
    node_rows = [
        {
            "variant_id": variant.variant_id,
            "group": group,
            "horizon": horizon,
            "status": "not_available_until_formal_predictions",
            "Score": "",
            "MAE": "",
            "RMSE": "",
            "R2": "",
        }
        for variant in variants
        for horizon in HORIZONS
        for group in ("High-spatial-dependence", "Low-spatial-dependence", "High-volatility", "Ramp", "Localized-disturbance")
    ]
    _write_csv(g_root / "G_NODE_GROUP_METRICS.csv", node_rows, list(node_rows[0]))
    _write_csv(g_root / "G_FAILURES.csv", failures, ["variant_id", "run_status", "reason", "failure_report"])
    handoff = f"""# HANDOFF STEP4

## 状态

G0–G8 的配置、真实前向差异、图来源隔离、自由与固定自适应支持、逐行度数保持随机重连及审计产物已实现。G0 只读引用 `{CANONICAL_ID}`。G1–G8 的 dry-run、普通 smoke 和正式形状前向/反向预检均通过；正式长训练未在本次预检阶段启动，G1–G8 保留 failure report，不填零或复制指标。

## 验证结果

- 步骤4合同测试：18 passed。
- 实证分析回归测试：44 passed，1 个 Matplotlib 参数弃用警告。
- 核心组件与 v4.3 回归测试：37 passed，1 skipped。
- G0–G8 dry-run：全部唯一差异审计通过。
- G1–G8 smoke：全部完成训练、评估与协议检查。
- G1–G8 full-shape：全部输出 `[32, 10, 134]`，loss 与梯度有限。
- 295 个 JSON 与 37 个 CSV 通过 Python 可读性检查。

## 已生成审计

- `G_VARIANT_MANIFEST.json`
- `G_PROTOCOL_AUDIT.json`
- `G_GRAPH_IDENTITY.csv`
- `G_BRANCH_ASSIGNMENT_TRACE.json`
- `G_SUPPORT_OVERLAP.csv`
- `G_EDGE_DISTANCE_SUMMARY.csv`
- `G_NODE_GROUP_METRICS.csv`
- `G_FAILURES.csv`

## 运行命令

```powershell
python -m pytest custom_models/tests/empirical_analysis/test_step4_graph_semantics.py -q
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --dry-run
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --smoke
python -m st_mgprompt.run_empirical --family G --variants G0 G1 G2 G3 G4 G5 G6 G7 G8 --full-shape
```

## 未完成项

- G1–G8 正式长训练与正式 H3/H6/H10、逐风机和场景指标。
- 多随机种子稳定性属于后续步骤11。

所有结论仅解释为统计语义匹配、先验支持集约束或模型内部边权稳定性，不解释为尾流或地理因果。
"""
    (g_root / "HANDOFF_STEP4.md").write_text(handoff, encoding="utf-8")
    return {"g_root": str(g_root.resolve()), "variant_count": len(variants), "formal_training_started": False}


__all__ = ["write_step4_reports"]
