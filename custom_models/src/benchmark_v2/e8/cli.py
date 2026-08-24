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
from .diagnostic_hooks import write_blocked_diagnostic_audit
from .evidence import build_evidence_manifest, build_missing_workbook_manifest
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


def _diagnostic_statuses(evidence: dict[str, Any], side: dict[str, Any] | None) -> dict[str, str]:
    internal = [row for row in evidence["evidence"] if row["evidence_id"].startswith("E8_INTERNAL_")]
    internal_ready = len(internal) == 5 and all(row["ready"] for row in internal)
    if not internal_ready:
        blocked = "BLOCKED_INTERNAL_PROTOCOL_MISMATCH"
        return {"representation": blocked, "cross": blocked, "macro": blocked, "st": blocked,
                "paired": "BLOCKED_PREDICTIONS_MISSING_OR_PROTOCOL_MISMATCH"}
    if side and side.get("status") == "PASS":
        return {"representation": "READY", "cross": "READY", "macro": "READY", "st": "READY", "paired": "READY"}
    return {"representation": "NOT_RUN", "cross": "NOT_RUN", "macro": "NOT_RUN", "st": "NOT_RUN", "paired": "NOT_RUN"}


def run_audit(*, original26_xlsx: str | None, analysis_root: str | Path = ANALYSIS_ROOT, doc_root: str | Path = DOC_ROOT) -> dict[str, Any]:
    root, docs = Path(analysis_root), Path(doc_root)
    root.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    protocol, config, grouped = audit_protocol(), audit_internal_config(), grouped_definition_audit()
    xlsx_path, discovery_errors = _resolve_xlsx(original26_xlsx)
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
    side_path = root / "E8_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json"
    side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.is_file() else None
    statuses = _diagnostic_statuses(evidence, side)
    readiness = build_readiness(
        evidence, representation_status=statuses["representation"], cross_status=statuses["cross"],
        macro_status=statuses["macro"], st_status=statuses["st"], paired_status=statuses["paired"],
        grouped_status=grouped["status"],
    )
    if side is None:
        side = write_blocked_diagnostic_audit(root, "Current-batch4 internal evidence is not 5/5 ready.")
    capability = build_capability_matrix()
    protocol["original26_audit"] = workbook_audit
    outputs = {
        "E8_EVIDENCE_MANIFEST.json": evidence, "E8_READINESS.json": readiness,
        "E8_PROTOCOL_AUDIT.json": protocol, "E8_INTERNAL_CONFIG_DIFF.json": config,
        "E8_MECHANISM_CAPABILITY_MATRIX.json": capability,
    }
    for name, payload in outputs.items():
        write_json(root / name, payload)
        write_json(docs / name, payload)
    write_json(docs / "E8_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json", side)
    write_csv(root / "E8_MECHANISM_CAPABILITY_MATRIX.csv", capability)
    write_csv(root / "e8_mechanism_capability_matrix.csv", capability)
    write_csv(docs / "E8_MECHANISM_CAPABILITY_MATRIX.csv", capability)
    write_group_gap(docs / "E8_GROUP_DEFINITION_GAP.md")
    manifest_path = write_artifact_manifest(root)
    write_json(docs / manifest_path.name, json.loads(manifest_path.read_text(encoding="utf-8")))
    return {
        "status": "PASS" if protocol["status"] == "PASS" else "FAIL", "readiness": readiness,
        "workbook_audit": workbook_audit, "analysis_root": str(root.resolve()), "doc_root": str(docs.resolve()),
    }


def run_diagnostics(*, analysis_root: str | Path = ANALYSIS_ROOT) -> dict[str, Any]:
    root = Path(analysis_root)
    evidence_path = root / "E8_EVIDENCE_MANIFEST.json"
    if not evidence_path.is_file():
        evidence = build_missing_workbook_manifest()
    else:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    internal = [row for row in evidence["evidence"] if row["evidence_id"].startswith("E8_INTERNAL_")]
    if len(internal) != 5 or not all(row["ready"] for row in internal):
        return write_blocked_diagnostic_audit(root, "BLOCKED_INTERNAL_PROTOCOL_MISMATCH: evaluate-only export is not allowed from the available batch32 artifacts.")
    return write_blocked_diagnostic_audit(root, "FORMAL_EVALUATE_ONLY_EXPORT_NOT_STARTED: invoke the programmatic hook/export adapter with the formal test dataloader.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e8", description="E8 read-only Prompt/Cross-Fusion audit and aggregation")
    parser.add_argument("--analysis-root", default=str(ANALYSIS_ROOT))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--original26-xlsx")
    commands.add_parser("diagnostics")
    for name in ("aggregate", "all"):
        command = commands.add_parser(name)
        command.add_argument("--original26-xlsx")
        command.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    root = Path(args.analysis_root)
    if args.command in {"readiness", "audit"}:
        print(json.dumps(run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root), ensure_ascii=False, indent=2))
        return 0
    if args.command == "diagnostics":
        result = run_diagnostics(analysis_root=root)
        write_json(DOC_ROOT / "E8_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json", result)
        write_artifact_manifest(root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "PASS" else 2
    result = run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root)
    if args.command == "all":
        result["diagnostics"] = run_diagnostics(analysis_root=root)
        result = run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root) | {"diagnostics": result["diagnostics"]}
    evidence = json.loads((root / "E8_EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    readiness = json.loads((root / "E8_READINESS.json").read_text(encoding="utf-8"))
    config = json.loads((root / "E8_INTERNAL_CONFIG_DIFF.json").read_text(encoding="utf-8"))
    try:
        result["aggregation"] = aggregate(evidence, readiness, config, output_root=root, require_complete=args.require_complete)
    except IncompleteEvidenceError as exc:
        print(json.dumps({"status": "BLOCKED_INCOMPLETE_EVIDENCE", "error": str(exc), "readiness": readiness}, ensure_ascii=False, indent=2))
        return 2
    write_artifact_manifest(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
