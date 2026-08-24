from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .aggregate import IncompleteEvidenceError, aggregate
from .capability_matrix import build_capability_matrix
from .config_audit import audit_internal_config
from .constants import ANALYSIS_ROOT, DOC_ROOT, PROJECT_ROOT
from .evidence import build_evidence_manifest, build_missing_workbook_manifest
from .graph_extractors import run_graph_diagnostics
from .grouped_analysis import grouped_definition_audit, write_group_gap
from .io_utils import write_csv, write_json
from .original26_loader import discover_original26, load_original26
from .protocol_audit import audit_protocol
from .readiness import build_readiness
from .report import write_artifact_manifest


def _resolve_xlsx(value: str | None) -> tuple[Path | None, list[str]]:
    if value:
        path = Path(value)
        return (path.resolve(), []) if path.is_file() else (None, ["MISSING_ORIGINAL26_XLSX"])
    return discover_original26(PROJECT_ROOT)


def run_audit(*, original26_xlsx: str | None, analysis_root: str | Path = ANALYSIS_ROOT, doc_root: str | Path = DOC_ROOT) -> dict[str, Any]:
    root, docs = Path(analysis_root), Path(doc_root)
    root.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    protocol = audit_protocol()
    config = audit_internal_config()
    grouped = grouped_definition_audit()
    xlsx_path, discovery_errors = _resolve_xlsx(original26_xlsx)
    workbook_audit: dict[str, Any]
    if xlsx_path is None:
        evidence = build_missing_workbook_manifest(discovery_errors[0])
        workbook_audit = {"path": None, "schema_valid": False, "errors": discovery_errors}
    else:
        try:
            workbook_models, workbook_audit = load_original26(xlsx_path)
            evidence = build_evidence_manifest(workbook_models)
        except Exception as exc:
            code = "ORIGINAL26_SCHEMA_OR_CONTENT_INVALID"
            evidence = build_missing_workbook_manifest(code)
            workbook_audit = {"path": str(xlsx_path), "schema_valid": False, "errors": [code, f"{type(exc).__name__}:{exc}"]}
    side_path = root / "E7_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json"
    if side_path.is_file():
        side = json.loads(side_path.read_text(encoding="utf-8"))
        diagnostics_status = "PARTIAL_EXTERNAL_ONLY" if side.get("mtgnn", {}).get("status") == "PASS" else "FAIL"
    else:
        side = {"schema_version": "e7_diagnostic_side_effect_audit_v1", "status": "NOT_RUN", "reason": "Run `e7 diagnostics`; internal diagnostics remain blocked by current batch4 evidence mismatch."}
        write_json(side_path, side)
        diagnostics_status = "NOT_RUN"
    readiness = build_readiness(
        evidence,
        grouped_status=grouped["status"],
        diagnostics_status=diagnostics_status,
        node_status="BLOCKED_PREDICTIONS_MISSING_OR_PROTOCOL_MISMATCH",
    )
    capability = build_capability_matrix()
    protocol["original26_audit"] = workbook_audit
    outputs = {
        "E7_EVIDENCE_MANIFEST.json": evidence,
        "E7_READINESS.json": readiness,
        "E7_PROTOCOL_AUDIT.json": protocol,
        "E7_INTERNAL_CONFIG_DIFF.json": config,
        "E7_GRAPH_CAPABILITY_MATRIX.json": capability,
    }
    for name, payload in outputs.items():
        write_json(root / name, payload)
        write_json(docs / name, payload)
    write_csv(root / "E7_GRAPH_CAPABILITY_MATRIX.csv", capability)
    write_csv(docs / "E7_GRAPH_CAPABILITY_MATRIX.csv", capability)
    write_group_gap(docs / "E7_GROUP_DEFINITION_GAP.md")
    manifest_path = write_artifact_manifest(root)
    write_json(docs / manifest_path.name, json.loads(manifest_path.read_text(encoding="utf-8")))
    return {"status": "PASS" if protocol["status"] == "PASS" else "FAIL", "readiness": readiness, "workbook_audit": workbook_audit, "analysis_root": str(root.resolve()), "doc_root": str(docs.resolve())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e7", description="E7 read-only graph mechanism audit and aggregation")
    parser.add_argument("--analysis-root", default=str(ANALYSIS_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "audit"):
        command = subparsers.add_parser(name)
        command.add_argument("--original26-xlsx")
    diagnostics = subparsers.add_parser("diagnostics")
    diagnostics.add_argument("--output-root")
    for name in ("aggregate", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--original26-xlsx")
        command.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    root = Path(args.analysis_root)
    if args.command in {"readiness", "audit"}:
        result = run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "diagnostics":
        result = run_graph_diagnostics(args.output_root or root)
        side_path = Path(args.output_root or root) / "E7_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json"
        if side_path.is_file():
            write_json(DOC_ROOT / side_path.name, json.loads(side_path.read_text(encoding="utf-8")))
        write_artifact_manifest(args.output_root or root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("side_effect_status") == "PASS" else 1
    result = run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root)
    if args.command == "all":
        diagnostic = run_graph_diagnostics(root)
        result = run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root)
        result["diagnostics"] = diagnostic
    evidence = json.loads((root / "E7_EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    readiness = json.loads((root / "E7_READINESS.json").read_text(encoding="utf-8"))
    try:
        aggregation = aggregate(evidence, readiness, output_root=root, require_complete=args.require_complete)
    except IncompleteEvidenceError as exc:
        print(json.dumps({"status": "BLOCKED_INCOMPLETE_EVIDENCE", "error": str(exc), "readiness": readiness}, ensure_ascii=False, indent=2))
        return 2
    result["aggregation"] = aggregation
    write_artifact_manifest(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
