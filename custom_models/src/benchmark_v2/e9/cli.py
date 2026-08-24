from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .aggregate import IncompleteE9EvidenceError, aggregate_complete
from .config_diff import build_config_diff_audit
from .constants import DOC_ROOT, LOSS_PROFILE, MODEL_IDS, PREFLIGHT_ROOT, TRANSFER_ROOT
from .evidence import build_evidence_manifest, build_transfer_readiness
from .io_utils import read_json, write_json
from .loss_contract import loss_input_contract, loss_profile_manifest, loss_state_manifest
from .original26_resolver import audit_control_artifacts, load_workbook_controls, resolve_original26
from .pairing_audit import build_pairing_audit, pairing_markdown
from .portability_audit import portability_audit
from .preflight import preflight_all
from .readiness import build_readiness
from .report import build_formal_report
from .training import formal_train_transfer
from .variants import run_id_map, variant_manifest


def _missing_control_audit(reason: str) -> dict[str, Any]:
    from .variants import CONTROL_RUN_IDS
    return {
        "schema_version": "e9_original26_control_audit_v1",
        "source_class": "ORIGINAL26_CONTROL", "control_count": 6, "ready_count": 0,
        "ORIGINAL_MASKED_MSE_REFERENCE_READY": "0/6",
        "controls": [
            {
                "model_id": model_id, "run_id": CONTROL_RUN_IDS[model_id], "source_root": None,
                "run_status": None, "protocol_status": None, "loss_id": None, "batch": None, "seed": None,
                "artifact_completeness": False, "effective_config": {}, "resolved_config": {}, "metrics": None,
                "checkpoint_metadata": {}, "validation_status": "FAIL", "validation_errors": [reason], "ready": False,
            }
            for model_id in MODEL_IDS
        ],
        "e5_consumed": False,
    }


def _blocked_config_diff(reason: str) -> dict[str, Any]:
    from .variants import CONTROL_RUN_IDS, canonical_transfer_run_id
    return {
        "schema_version": "e9_loss_only_config_diff_v1", "ready_count": 0, "LOSS_ONLY_CONFIG_READY": "0/6",
        "pairs": [
            {
                "model_id": model_id, "control_run_id": CONTROL_RUN_IDS[model_id],
                "transfer_run_id": canonical_transfer_run_id(model_id), "allowed_differences": [],
                "unexpected_differences": [{"path": "control", "reason": reason}],
                "model_source_match": False, "model_architecture_match": False,
                "model_hyperparameter_match": False, "data_protocol_match": False,
                "training_protocol_match": False, "graph_protocol_match": False,
                "loss_identity_changed": False, "loss_only_diff_valid": False, "pair_status": "INVALID_CONFIG_DIFF",
            }
            for model_id in MODEL_IDS
        ],
    }


def _write_dual(name: str, payload: Any, docs: Path, root: Path) -> None:
    write_json(docs / name, payload)
    write_json(root / name, payload)


def run_audit(*, original26_xlsx: str | None, doc_root: str | Path = DOC_ROOT, analysis_root: str | Path = TRANSFER_ROOT) -> dict[str, Any]:
    docs, root = Path(doc_root), Path(analysis_root)
    docs.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    portability = portability_audit()
    contract = loss_input_contract()
    profile = loss_profile_manifest(LOSS_PROFILE)
    state = loss_state_manifest()
    variants = variant_manifest()
    run_map = run_id_map()
    workbook_path, discovery_errors = resolve_original26(original26_xlsx)
    workbook_audit: dict[str, Any]
    if workbook_path is None:
        reason = discovery_errors[0]
        control_audit = _missing_control_audit(reason)
        workbook_audit = {"path": None, "schema_valid": False, "errors": discovery_errors}
        config_diff = _blocked_config_diff(reason)
    else:
        try:
            workbook_controls, workbook_audit = load_workbook_controls(workbook_path)
            control_audit = audit_control_artifacts(workbook_controls)
            control_audit["workbook_audit"] = workbook_audit
            config_diff = build_config_diff_audit(control_audit, root)
        except Exception as exc:
            reason = f"ORIGINAL26_SCHEMA_OR_CONTENT_INVALID:{type(exc).__name__}:{exc}"
            control_audit = _missing_control_audit(reason)
            workbook_audit = {"path": str(workbook_path), "schema_valid": False, "errors": [reason]}
            config_diff = _blocked_config_diff(reason)
    evidence = build_evidence_manifest(control_audit, root)
    pairing = build_pairing_audit(control_audit, evidence.get("transfer_by_model"))
    readiness = build_readiness(portability, evidence)
    preflight_path = Path(PREFLIGHT_ROOT) / "E9_PREFLIGHT_INVENTORY.json"
    preflight = read_json(preflight_path) if preflight_path.is_file() else {
        "schema_version": "e9_preflight_inventory_v1", "inventory": [
            {"model_id": model_id, "status": "NOT_RUN", "formal_training_started": False} for model_id in MODEL_IDS
        ], "passed": 0, "expected": 6, "all_passed": False, "formal_training_started": False,
    }
    outputs = {
        "E9_PORTABILITY_AUDIT.json": portability,
        "E9_LOSS_INPUT_CONTRACT.json": contract,
        "E9_LOSS_PROFILE.json": profile,
        "E9_LOSS_STATE_MANIFEST.json": state,
        "E9_VARIANT_MANIFEST.json": variants,
        "E9_RUN_ID_MAP.json": run_map,
        "E9_ORIGINAL26_CONTROL_AUDIT.json": control_audit,
        "E9_LOSS_ONLY_CONFIG_DIFF.json": config_diff,
        "E9_PAIRING_AUDIT.json": pairing,
        "E9_PREFLIGHT_INVENTORY.json": preflight,
        "E9_EVIDENCE_MANIFEST.json": {key: value for key, value in evidence.items() if key != "transfer_by_model"},
        "E9_READINESS.json": readiness,
    }
    for name, payload in outputs.items():
        _write_dual(name, payload, docs, root)
    (docs / "E9_PAIRING_AUDIT.md").write_text(pairing_markdown(pairing), encoding="utf-8")
    return {
        "status": "PASS" if portability["e9_b_allowed"] else "BLOCKED_PORTABILITY",
        "workbook_audit": workbook_audit, "readiness": readiness,
        "doc_root": str(docs.resolve()), "analysis_root": str(root.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark_v2.e9")
    parser.add_argument("--analysis-root", default=str(TRANSFER_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "readiness", "precheck"):
        command = sub.add_parser(name)
        command.add_argument("--original26-xlsx")
    preflight_parser = sub.add_parser("preflight-all")
    preflight_parser.add_argument("--device", default="cuda")
    preflight_parser.add_argument("--preflight-root", default=str(PREFLIGHT_ROOT))
    train = sub.add_parser("train-one")
    train.add_argument("--model", required=True, choices=MODEL_IDS)
    train.add_argument("--input-path")
    train.add_argument("--target-path")
    train.add_argument("--device", default="cuda")
    train.add_argument("--preflight-root", default=str(PREFLIGHT_ROOT))
    sub.add_parser("transfer-readiness")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--original26-xlsx")
    aggregate.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    root = Path(args.analysis_root)
    try:
        if args.command in {"audit", "readiness", "precheck"}:
            result = run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 2
        if args.command == "preflight-all":
            result = preflight_all(device=args.device, output_root=args.preflight_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["all_passed"] else 3
        if args.command == "train-one":
            kwargs = {"output_root": root, "device": args.device, "preflight_root": args.preflight_root}
            if args.input_path:
                kwargs["input_path"] = args.input_path
            if args.target_path:
                kwargs["target_path"] = args.target_path
            result = formal_train_transfer(args.model, **kwargs)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "transfer-readiness":
            result = build_transfer_readiness(root)
            write_json(root / "E9_TRANSFER_READINESS.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["MSMG_DWU_TRANSFER_READY_BOOL"] else 2
        if args.command == "aggregate":
            audit_result = run_audit(original26_xlsx=args.original26_xlsx, analysis_root=root)
            readiness = read_json(root / "E9_READINESS.json")
            evidence = read_json(root / "E9_EVIDENCE_MANIFEST.json")
            result = aggregate_complete(evidence, readiness, output_root=root, require_complete=args.require_complete)
            payload_path = root / "e9_report_payload.json"
            payload = read_json(payload_path)
            payload["audit"] = {
                "portability": read_json(root / "E9_PORTABILITY_AUDIT.json"),
                "loss_contract": read_json(root / "E9_LOSS_INPUT_CONTRACT.json"),
                "pairing": read_json(root / "E9_PAIRING_AUDIT.json"),
                "config_diff": read_json(root / "E9_LOSS_ONLY_CONFIG_DIFF.json"),
                "preflight": read_json(root / "E9_PREFLIGHT_INVENTORY.json"),
                "artifact_manifest": {"artifacts": []},
            }
            write_json(payload_path, payload)
            result["report"] = build_formal_report(root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    except IncompleteE9EvidenceError as exc:
        readiness = read_json(root / "E9_READINESS.json") if (root / "E9_READINESS.json").is_file() else {}
        print(json.dumps({"status": "BLOCKED_INCOMPLETE_EVIDENCE", "error": str(exc), "readiness": readiness}, ensure_ascii=False, indent=2))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, ensure_ascii=False, indent=2))
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
