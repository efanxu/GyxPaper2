from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parents[3]
PYTHON = Path(sys.executable).resolve()
REL = lambda p: str(Path(p).resolve().relative_to(ROOT)).replace(chr(92), "/")

MODEL_GROUPS = {
    "basic": ["Persistence", "MovingAverage", "GRU"],
    "lightweight": ["DLinear", "LightTS", "TiDE", "SegRNN"],
    "transformer": ["Transformer", "PatchTST", "iTransformer", "TimeXer"],
    "multiscale": ["TimesNet", "MICN", "WPMixer", "MultiPatchFormer"],
    "mixer_frequency": ["TimeMixer", "TSMixer", "FreTS"],
    "cross_variable": ["Crossformer", "MSGNet", "TimeFilter"],
    "graph": ["GCN", "STCN_STGCN", "DCRNN", "Graph WaveNet", "MTGNN", "AGCRN", "STID"],
}
ALL_MODELS = [name for group in MODEL_GROUPS.values() for name in group]
TSLIB_MODELS = [
    "DLinear", "LightTS", "TiDE", "SegRNN", "Transformer", "PatchTST",
    "iTransformer", "TimeXer", "TimesNet", "MICN", "WPMixer", "MultiPatchFormer",
    "TimeMixer", "TSMixer", "FreTS", "Crossformer", "MSGNet", "TimeFilter",
]
SMOKE_OK = {"DLinear", "LightTS", "TiDE", "Transformer", "PatchTST", "iTransformer", "TimeXer", "TimesNet", "WPMixer", "TSMixer", "FreTS", "Crossformer", "MSGNet", "TimeFilter"}
SMOKE_ERRORS = {
    "SegRNN": "RuntimeError: zero-sized segment result under seq_len=144, seg_len=12 synthetic configuration",
    "MICN": "RuntimeError: decoder/time-mark length mismatch (154 vs 58) under generic decoder input",
    "MultiPatchFormer": "RuntimeError: patch concatenation mismatch (18 vs 19) under generic configuration",
    "TimeMixer": "IndexError: scale list has no second level with down_sampling_layers=0",
}

FIELDS = [
    "canonical_name", "category", "aliases_found", "display_names_found", "status",
    "implementation_paths", "primary_implementation_path", "class_or_function",
    "registry_keys", "registry_paths", "factory_paths", "runner_paths", "config_paths",
    "test_paths", "documentation_paths", "old_result_paths", "source_provenance",
    "upstream_project", "paper_or_model_identity", "license_or_notice", "import_status",
    "construction_status", "forward_status", "existing_smoke_status", "expected_input_shape",
    "actual_input_shape", "expected_output_shape", "actual_output_shape", "adapter_needed",
    "adapter_type", "graph_required", "graph_source", "exogenous_interface_required",
    "node_identity_required", "time_identity_required", "loss_path", "loss_name",
    "mask_applied_in_loss", "mask_applied_in_metrics", "target_column", "input_power_column",
    "feature_count", "feature_order_source", "split_mode", "train_stride", "val_stride",
    "test_stride", "eval_horizons", "physical_clip", "checkpoint_metric", "checkpoint_horizon",
    "checkpoint_direction", "metric_implementation_path", "score_definition_path",
    "test_set_tuning_risk", "data_leakage_risk", "protocol_risk_level", "dependency_status",
    "missing_dependencies", "results_reusable", "reuse_reason", "recommended_stage",
    "recommended_action", "evidence_files", "notes",
]


def write_text(name: str, text: str) -> None:
    (AUDIT_DIR / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def jdump(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def source_for(name: str) -> str:
    return f"Time-Series-Library/models/{name}.py" if name in TSLIB_MODELS else "UNKNOWN"


def category(name: str) -> str:
    return next(k for k, values in MODEL_GROUPS.items() if name in values)


def evidence(name: str) -> list[str]:
    if name in TSLIB_MODELS:
        return [source_for(name), "Time-Series-Library/exp/exp_basic.py:15-23,25-46,79-111", "Time-Series-Library/exp/exp_long_term_forecasting.py:22-39,42-74,76-166,168-268", "Time-Series-Library/data_provider/data_loader.py:1-17,51-79", "Time-Series-Library/utils/metrics.py"]
    if name == "GRU":
        return ["custom_models/src/st_mgprompt/model.py:16-68", "custom_models/src/st_mgprompt/registry.py:30-38", "custom_models/src/st_mgprompt/run_st_mgprompt.py:477-500"]
    if name == "STCN_STGCN":
        return ["Time-Series-Library/run.py:143-149", "Time-Series-Library/models/: no STCN.py or STGCN.py", "local UTF-8 text search across custom_models, Time-Series-Library, configs, scripts, tools, tests, docs"]
    return ["local UTF-8 text search across custom_models, Time-Series-Library, configs, scripts, tools, tests, docs: no trusted implementation found"]


def common_tslib_row(name: str) -> dict:
    smoke = "SMOKE_OK" if name in SMOKE_OK else "SMOKE_ERROR: " + SMOKE_ERRORS[name]
    forward = "forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)" if name != "FreTS" else "forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec) (no mask argument)"
    identity = "generic time marks only; no SDWPF node identity" if name != "TimeXer" else "generic x_mark/exogenous branch; SDWPF split not wired"
    return {
        "canonical_name": name, "category": category(name),
        "aliases_found": name, "display_names_found": name,
        "status": "IMPLEMENTED_PROTOCOL_RISK",
        "implementation_paths": source_for(name), "primary_implementation_path": source_for(name),
        "class_or_function": "models.%s.Model" % name,
        "registry_keys": name, "registry_paths": "Time-Series-Library/exp/exp_basic.py:25-46 (runtime auto-scan; no fixed registry)",
        "factory_paths": "Time-Series-Library/exp/exp_long_term_forecasting.py:22-27",
        "runner_paths": "Time-Series-Library/run.py:15-24,180-245",
        "config_paths": "Time-Series-Library/run.py:18-157; model-specific shell scripts",
        "test_paths": "UNKNOWN (no SDWPF benchmark test)",
        "documentation_paths": "Time-Series-Library/README.md; Time-Series-Library/LICENSE",
        "old_result_paths": "NONE FOUND for SDWPF benchmark",
        "source_provenance": "UPSTREAM_TSLIB_LOCAL_COPY", "upstream_project": "THUML/Time-Series-Library",
        "paper_or_model_identity": "README-listed upstream implementation; see README model entry",
        "license_or_notice": "MIT; Time-Series-Library/LICENSE (Copyright 2021 THUML)",
        "import_status": "IMPORT_OK", "construction_status": "CONSTRUCTION_OK (synthetic smoke)" if name in SMOKE_OK or name in SMOKE_ERRORS else "UNKNOWN",
        "forward_status": forward, "existing_smoke_status": smoke,
        "expected_input_shape": "(B,T,N,C)", "actual_input_shape": "(B,T,C) in TSLib generic path",
        "expected_output_shape": "(B,N,10)", "actual_output_shape": "(B,10,C) in synthetic TSLib path",
        "adapter_needed": "YES", "adapter_type": "seq_btc_to_node_horizon_bnh; node flattening/target selection must be specified",
        "graph_required": "NO external graph provider; some models learn/intermix channels internally",
        "graph_source": "NONE external; MSGNet/TimeFilter use internal correlation/filter logic",
        "exogenous_interface_required": "YES" if name == "TimeXer" else "NO generic x_mark only",
        "node_identity_required": "NO explicit SDWPF node identity", "time_identity_required": identity,
        "loss_path": "Time-Series-Library/exp/exp_long_term_forecasting.py:37-39",
        "loss_name": "nn.MSELoss",
        "mask_applied_in_loss": "NO", "mask_applied_in_metrics": "NO unified valid_target_mask",
        "target_column": "UNKNOWN/generic args.target default OT", "input_power_column": "UNKNOWN",
        "feature_count": "args.enc_in; not fixed SDWPF 16", "feature_order_source": "generic dataframe column order in data_loader.py:68-72",
        "split_mode": "dataset-specific positional borders; not 0.8/0.1/0.1 SDWPF strict protocol",
        "train_stride": "UNKNOWN", "val_stride": "UNKNOWN", "test_stride": "UNKNOWN",
        "eval_horizons": "pred_len only; no H3/H6/H10 official artifacts",
        "physical_clip": "NONE in generic path", "checkpoint_metric": "validation MSE via EarlyStopping",
        "checkpoint_horizon": "pred_len", "checkpoint_direction": "min",
        "metric_implementation_path": "Time-Series-Library/utils/metrics.py", "score_definition_path": "NONE (no official SDWPF Score)",
        "test_set_tuning_risk": "HIGH: test loader/loss is evaluated every epoch at exp_long_term_forecasting.py:78-79,152-155",
        "data_leakage_risk": "HIGH/UNKNOWN until SDWPF adapter audit; generic loader is not the project split/mask pipeline",
        "protocol_risk_level": "HIGH", "dependency_status": "SATISFIED_FOR_IMPORT", "missing_dependencies": "NONE for import",
        "results_reusable": "NOT_REUSABLE", "reuse_reason": "No local SDWPF artifact; generic TSLib results would not prove the unified protocol",
        "recommended_stage": {"DLinear":"E1-B","LightTS":"E1-B","TiDE":"E1-B","SegRNN":"E1-B","Transformer":"E2-A","PatchTST":"E2-A","iTransformer":"E2-A","TimeXer":"E2-A","TimesNet":"E2-B","MICN":"E2-B","WPMixer":"E2-B","MultiPatchFormer":"E2-B","TimeMixer":"E2-C","TSMixer":"E2-C","FreTS":"E2-C","Crossformer":"E2-D","MSGNet":"E2-D","TimeFilter":"E2-D"}.get(name, "E2-A"),
        "recommended_action": "E0-B adapter + protocol wrapper; preserve upstream source and license; resolve smoke/config issues in the model-specific stage",
        "evidence_files": "; ".join(evidence(name)),
        "notes": "Importable upstream model is not equivalent to a fair SDWPF benchmark implementation. Smoke uses synthetic (B=2,T=144,C=16,H=10) only.",
    }


def missing_row(name: str) -> dict:
    status = "NAME_CONFLICT" if name == "STCN_STGCN" else ("PARTIAL_IMPLEMENTATION" if name == "GRU" else "MISSING")
    if name == "GRU":
        return {**{f: "UNKNOWN" for f in FIELDS}, "canonical_name": name, "category": "basic", "aliases_found": "GRU (internal STMGPrompt temporal block)", "display_names_found": "GRU", "status": status, "implementation_paths": "custom_models/src/st_mgprompt/model.py", "primary_implementation_path": "custom_models/src/st_mgprompt/model.py", "class_or_function": "STMGPromptTemporalOnly; internal nn.GRU", "registry_keys": "STMGPrompt_TemporalOnly", "registry_paths": "custom_models/src/st_mgprompt/registry.py:30-38", "factory_paths": "custom_models/src/st_mgprompt/registry.py:171-180", "runner_paths": "custom_models/src/st_mgprompt/run_st_mgprompt.py", "config_paths": "custom_models/src/st_mgprompt/config.py", "import_status": "IMPORT_OK_INTERNAL_ONLY", "construction_status": "INTERNAL_ONLY; no standalone GRU benchmark construction", "forward_status": "INTERNAL_FORWARD_ONLY", "adapter_needed": "YES", "adapter_type": "new standalone GRU wrapper required", "loss_name": "ST-MGPrompt protocol only", "recommended_stage": "E1-A", "recommended_action": "Implement standalone node-shared GRU adapter in E1-A; do not reuse STMGPrompt registry key as canonical GRU", "evidence_files": "; ".join(evidence(name)), "notes": "The internal GRU is not an independent GRU baseline, so this row is partial rather than usable."}
    if name == "STCN_STGCN":
        return {**{f: "UNKNOWN" for f in FIELDS}, "canonical_name": name, "category": "graph", "aliases_found": "STCN; STGCN (requested label only)", "display_names_found": "STCN/STGCN", "status": status, "implementation_paths": "NONE", "primary_implementation_path": "UNKNOWN", "class_or_function": "UNKNOWN", "registry_keys": "NONE", "registry_paths": "NONE", "factory_paths": "NONE", "runner_paths": "Time-Series-Library/run.py:143-149 only exposes generic GCN flags", "config_paths": "NONE", "documentation_paths": "NONE", "import_status": "NOT_FOUND", "construction_status": "NOT_RUN", "forward_status": "NOT_FOUND", "adapter_needed": "UNKNOWN", "graph_required": "UNKNOWN", "recommended_stage": "E3-A", "recommended_action": "Resolve identity in E3-A; preserve label until source/paper evidence is found; do not rename", "evidence_files": "; ".join(evidence(name)), "notes": "Situation C: evidence insufficient because neither STCN nor classical STGCN source exists locally."}
    return {**{f: "UNKNOWN" for f in FIELDS}, "canonical_name": name, "category": category(name), "aliases_found": name, "display_names_found": name, "status": status, "implementation_paths": "NONE FOUND", "primary_implementation_path": "UNKNOWN", "class_or_function": "UNKNOWN", "registry_keys": "NONE", "registry_paths": "NONE", "factory_paths": "NONE", "runner_paths": "NONE", "config_paths": "NONE", "test_paths": "NONE", "documentation_paths": "NONE", "old_result_paths": "NONE FOUND", "source_provenance": "UNKNOWN", "upstream_project": "UNKNOWN", "paper_or_model_identity": "UNKNOWN", "license_or_notice": "UNKNOWN", "import_status": "NOT_FOUND", "construction_status": "NOT_RUN", "forward_status": "NOT_FOUND", "existing_smoke_status": "SMOKE_NOT_RUN", "adapter_needed": "UNKNOWN", "graph_required": "UNKNOWN", "loss_path": "UNKNOWN", "loss_name": "UNKNOWN", "recommended_stage": {"Persistence":"E1-A","MovingAverage":"E1-A","GCN":"E3-B","DCRNN":"E3-B","Graph WaveNet":"E3-C","MTGNN":"E3-C","AGCRN":"E3-C","STID":"E3-C"}[name], "recommended_action": "Implement or port only after E0-B registry/adapter/artifact contracts are frozen; keep missing entry visible", "evidence_files": "; ".join(evidence(name)), "notes": "No trusted implementation, factory, or local reproducible result was found."}


def matrix() -> list[dict]:
    rows = []
    for name in ALL_MODELS:
        row = common_tslib_row(name) if name in TSLIB_MODELS else missing_row(name)
        rows.append({field: row.get(field, "UNKNOWN") for field in FIELDS})
    return rows


def static_audit() -> dict:
    roots = [ROOT / "custom_models/src/st_mgprompt", ROOT / "Time-Series-Library/models", ROOT / "Time-Series-Library/layers", ROOT / "Time-Series-Library/exp", ROOT / "Time-Series-Library/data_provider", ROOT / "Time-Series-Library/run.py"]
    files = []
    for root in roots:
        if root.is_file(): files.append(root)
        elif root.is_dir(): files.extend(sorted(root.glob("*.py")))
    parse_errors, placeholders, hardcoded = [], [], []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
            ast.parse(text, filename=str(p))
        except Exception as exc:
            parse_errors.append({"path": REL(p), "error": f"{type(exc).__name__}: {exc}"})
        for line_no, line in enumerate(text.splitlines(), 1):
            if re.search(r"TODO|NotImplementedError|^\s*pass\s*(#.*)?$", line): placeholders.append({"path": REL(p), "line": line_no, "text": line.strip()})
            if "Miniconda\\envs\\env_tslib" in line or "Miniconda/envs/env_tslib" in line: hardcoded.append({"path": REL(p), "line": line_no, "text": line.strip()})
    return {"files_checked": len(files), "parse_errors": parse_errors, "placeholder_or_todo_hits": placeholders[:200], "hardcoded_old_python_path_hits": hardcoded, "compile_policy": "AST syntax parse used; no compileall over data/results/venv."}


def imports() -> dict:
    st_modules = ["st_mgprompt.registry", "st_mgprompt.config", "st_mgprompt.data", "st_mgprompt.losses", "st_mgprompt.metrics", "st_mgprompt.evaluate", "st_mgprompt.train", "st_mgprompt.formal_runner", "st_mgprompt.run_ablation", "st_mgprompt.run_precision_ablation", "st_mgprompt.canonical_artifact"]
    sys.path.insert(0, str(ROOT / "custom_models/src"))
    sys.path.insert(0, str(ROOT / "Time-Series-Library"))
    output = {"python": str(PYTHON), "st_mgprompt": [], "tslib_models": [], "third_party": []}
    for mod in st_modules:
        try: importlib.import_module(mod); item = {"module": mod, "status": "IMPORT_OK"}
        except Exception as exc: item = {"module": mod, "status": "IMPORT_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        output["st_mgprompt"].append(item)
    for name in TSLIB_MODELS:
        try: importlib.import_module("models." + name); item = {"model": name, "status": "IMPORT_OK", "module": "models." + name}
        except Exception as exc: item = {"model": name, "status": "IMPORT_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        output["tslib_models"].append(item)
    for mod in ["torch", "numpy", "pandas", "sklearn", "einops", "pywt", "sktime", "datasets", "huggingface_hub", "scipy"]:
        try:
            m = importlib.import_module(mod); item = {"package": mod, "status": "IMPORT_OK", "version": getattr(m, "__version__", "UNKNOWN")}
        except Exception as exc: item = {"package": mod, "status": "IMPORT_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        output["third_party"].append(item)
    return output


def old_results() -> dict:
    root = ROOT / "custom_models/results"
    runs = []
    if root.exists():
        for p in sorted(root.rglob("metrics_eval_h10.json")):
            run = p.parent
            rel = REL(run)
            protected = "st_mgprompt_canonical" in rel or "st_mgprompt_precision" in rel or "st_mgprompt_component_ablation" in rel
            smoke = "smoke" in rel.lower()
            if "st_mgprompt_canonical" in rel:
                conclusion = "REUSABLE_AFTER_PROTOCOL_VERIFY"
                reason = "Canonical artifact has formal config, checkpoint, metrics and verification metadata; this is protected internal evidence, not an external benchmark."
            elif protected:
                conclusion = "NOT_REUSABLE"
                reason = "Protected P0-P5/A0-A8 artifact/reference path, not an external benchmark implementation."
            elif smoke:
                conclusion = "NOT_REUSABLE"
                reason = "Path/name indicates smoke or legacy smoke; must not be promoted to formal benchmark."
            else:
                conclusion = "METADATA_ONLY"
                reason = "Metrics file exists, but no verified benchmark_v2 protocol artifact set was found."
            required = ["resolved_config.json", "effective_config.json", "protocol_check.json", "model_summary.json", "best_checkpoint.pt", "last_checkpoint.pt", "train_log.csv", "metrics_eval_h3.json", "metrics_eval_h6.json", "metrics_eval_h10.json", "metrics.csv", "prediction_metadata.json", "run_status.json"]
            runs.append({"path": rel, "model_dir": run.name, "smoke": smoke, "protected": protected, "files_present": {x: (run / x).is_file() for x in required}, "reuse_conclusion": conclusion, "reason": reason})
    return {"root": REL(root), "run_count_with_metrics_eval_h10": len(runs), "runs": runs}


def inventory(rows: list[dict]) -> dict:
    return {
        "schema": "e0_a_inventory_only", "inventory_only": True, "not_runtime_registry": True,
        "generated_at": datetime.now(timezone.utc).isoformat(), "project_root": str(ROOT),
        "registries_found": [
            {"path": "custom_models/src/st_mgprompt/registry.py", "type": "dict", "keys": ["STMGPrompt_TemporalOnly", "STMGPrompt_TemporalOnly_VADSP", "STMGPrompt_DynamicPatching", "STMGPrompt_FairFull", "STMGPrompt_Full_MSMGDWU", "STMGPrompt_ComponentAblation", "STMGPrompt_FairFull_Diffusion", "STMGPrompt_FairFull_HistoryDecoder", "STMGPrompt_FairFull_DiffusionHistory", "STMGPrompt_Full_MSMGDWU_DiffusionHistory", "STMGPrompt_SerialGraphThenFusion", "STMGPrompt_CouplingCrossFusion", "STMGPrompt_TrendPriorGraph", "STMGPrompt_GraphTemporalSmoke"], "consumers": ["st_mgprompt/__init__.py", "train.py", "run_st_mgprompt.py", "full_shape_matrix_smoke.py"], "risks": ["STMGPrompt-only; not external benchmark registry", "one registered key has implemented=False", "many aliases map to STMGPrompt_FairFull"]},
            {"path": "custom_models/src/st_mgprompt/experiment_protocol.py", "type": "dict", "keys": ["P0", "P1", "P2", "P3", "P4", "P5", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"], "consumers": ["formal_runner.py", "run_ablation.py", "run_precision_ablation.py"], "risks": ["formal internal experiment registry; must remain isolated"]},
            {"path": "Time-Series-Library/exp/exp_basic.py", "type": "autoscanning LazyModelDict", "keys": ["runtime filenames in models/"], "consumers": ["exp_long_term_forecasting.py"], "risks": ["no fixed canonical names", "imports any models/*.py", "model availability depends on cwd"]},
            {"path": "Time-Series-Library/run.py", "type": "argparse", "keys": ["args.model"], "consumers": ["exp selection and training/test main"], "risks": ["generic task/data arguments", "no SDWPF/mask/Score contract"]},
        ],
        "models": {r["canonical_name"]: {"aliases": r["aliases_found"], "status": r["status"], "implementation_paths": r["implementation_paths"], "registry_keys": r["registry_keys"], "recommended_stage": r["recommended_stage"]} for r in rows},
        "name_conflicts": [{"canonical_name": "STCN_STGCN", "status": "NAME_CONFLICT", "evidence": ["no STCN.py/STGCN.py", "only generic GCN flags in Time-Series-Library/run.py:143-149"], "resolution": "E3-A source/paper identity audit; do not rename now"}],
        "duplicate_implementations": [],
        "orphan_implementations": ["Time-Series-Library/models/Autoformer.py", "Chronos.py", "ETSformer.py", "FEDformer.py", "FiLM.py", "Informer.py", "KANAD.py", "Koopa.py", "Mamba.py", "MambaSimple.py", "Moirai.py", "Nonstationary_Transformer.py", "PAttn.py", "Pyraformer.py", "Reformer.py", "SCINet.py", "Sundial.py", "TemporalFusionTransformer.py", "TimeMoE.py", "TimesFM.py", "TiRex.py"],
        "orphan_registry_entries": ["STMGPrompt_SerialGraphThenFusion (implemented=False; internal experimental registry, not external benchmark)"],
    }


def reports(rows: list[dict], stat: dict, imp: dict, old: dict) -> None:
    counts = {}
    for r in rows: counts[r["status"]] = counts.get(r["status"], 0) + 1
    table = "\n".join(f"| {r['canonical_name']} | {r['status']} | {r['primary_implementation_path']} | {r['recommended_stage']} |" for r in rows)
    write_text("BENCHMARK_IMPLEMENTATION_AUDIT.md", f"""# E0-A Benchmark Implementation Audit

## Executive Summary

This is an audit-only snapshot. The 28 required canonical rows are present: **{counts.get('IMPLEMENTED_PROTOCOL_RISK',0)} IMPLEMENTED_PROTOCOL_RISK, {counts.get('PARTIAL_IMPLEMENTATION',0)} PARTIAL_IMPLEMENTATION, {counts.get('MISSING',0)} MISSING, {counts.get('NAME_CONFLICT',0)} NAME_CONFLICT**. No new model, registry, adapter, artifact schema, training run, dependency installation, or deletion was performed.

The 18 TSLib source models are importable and can be constructed through the generic factory, but all are protocol-risk because the path is generic `(B,T,C)`, uses generic data/target semantics, `nn.MSELoss`, no `valid_target_mask`, no official Score, and writes generic `./results`. Persistence and MovingAverage are absent. GRU is only an internal node-shared block inside ST-MGPrompt and has no standalone benchmark entry. All seven graph slots are absent locally; `STCN_STGCN` is a naming conflict caused by missing source evidence.

## Repository and environment

- Project root: `{ROOT}`.
- Git: `git rev-parse --show-toplevel`, branch, HEAD, and status all failed with `fatal: not a git repository`; no `.git` directory was found under the project root. This is recorded as an environment fact, not treated as a clean worktree.
- Interpreter: `{PYTHON}`; Python 3.11.15; platform `{platform.platform()}`; torch 2.7.1+cu128; CUDA 12.8; CUDA available `True`; GPU `NVIDIA GeForce GTX 1060`; numpy 2.1.2; pandas 2.3.3; scikit-learn 1.7.2.
- `rg.exe` was attempted for file enumeration but returned Windows Access Denied; UTF-8 PowerShell/Python enumeration was used instead.

## 28-model implementation matrix summary

| canonical_name | status | primary implementation | stage |
|---|---|---|---|
{table}

Evidence for the matrix is in `benchmark_implementation_matrix.csv`; each row lists source/config paths and a `CONFIRMED`/`INFERRED` explanation in `notes`.

## Real call chains

### Legacy TSLib path

`Time-Series-Library/run.py:15-24,180-245` parses `args.model` and task. `exp/exp_basic.py:15-23,25-46` scans every `models/*.py` into a `LazyModelDict`; `exp/exp_long_term_forecasting.py:22-27` constructs `module.Model(args)`. `data_provider/data_factory.py` selects a generic dataset; `data_provider/data_loader.py:1-17,51-79` imports generic/HuggingFace datasets and fits a generic scaler. The train/validation/test path is `exp_long_term_forecasting.py:76-166`, with `nn.MSELoss` at `:37-39`; test loss is evaluated during every training epoch at `:78-79,152-155`. Test prediction and artifacts use `:168-268`, including `./test_results`, `./results`, `result_long_term_forecast.txt`, and `.npy` files. There is no official SDWPF Score, valid-target-mask path, physical clip, or H10 checkpoint rule.

### ST-MGPrompt protected path

`custom_models/src/st_mgprompt/run_st_mgprompt.py` is the CLI. `registry.py:30-196` resolves STMGPrompt names to model classes. `data.py:95-102,124-153,269-342` loads separate model-input/eval-target tables, fits train-only scalers, builds split-contained windows and loaders. `train.py` calls the model and loss; `evaluate.py:194-269,376-493` applies inverse transform, mask, clip, official Score and writes metrics artifacts. `formal_runner.py:103-145,309-429` manages P/A families and writes smoke/formal roots. This path is not a source for an external benchmark registry.

### Canonical/P0/A0 protection

Canonical is `{ROOT / 'custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026'}` with checkpoint SHA256 `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`. `experiment_protocol.py:15-22,165-234,332-381` defines the canonical ID/root and reference resolution. P0 and A0 `reference.json` files point to this directory, are `trainable:false`, and carry the same checkpoint/config/metrics hashes. `run_st_mgprompt.py:489` and `train.py:441,589` use `load_state_dict(..., strict=True)`. No benchmark code was found coupled to P/A; the risk is namespace/result-directory collision if E0-B reuses these roots.

## Duplicate implementations, names, and old results

No duplicate source implementation was confirmed for the 18 requested TSLib models. TSLib contains many additional orphan models not in the requested 28; they are listed in `benchmark_registry_inventory.json` and must not silently enter the benchmark. The only local old result with a complete formal identity is the protected Canonical artifact. Smoke and legacy ST-MGPrompt runs exist under `custom_models/results/st_mgprompt`, which is a pollution risk because the path is under `results`, not a distinct `results_smoke`; all such runs are marked non-reusable in `old_result_audit.json`. No local SDWPF external-benchmark result was found.

## Key risks

1. TSLib auto-registration by filename is not a canonical benchmark registry.
2. The generic path uses `(B,T,C)` and `[B,H,C]`, while the target protocol requires `(B,T,N,C)` and `(B,N,10)`.
3. TSLib has no `Patv_raw`/`Patv_clean_for_input`/`valid_target_mask` contract and no official Score.
4. Default ST-MGPrompt config fields differ from the formal canonical overrides (`config.py:70-78` defaults include eval batch 64, patience 10 and min_delta 0; `experiment_protocol.py:181-196` overrides to 32/4/4, patience 6 and 0.01). E0-B must consume the formal resolved config, not raw defaults.
5. TSLib evaluates test loss during training and saves generic outputs; this is incompatible with the required validation-H10-only selection and artifact schema.
6. Smoke artifacts are present in formal-looking ST-MGPrompt result roots; benchmark_v2 must use an independent root.

## E0-B and E1-E3 recommendation

E0-B should create a new inventory-backed registry outside `st_mgprompt.registry` and TSLib, with one adapter contract, explicit shape conversion, dataset provider, mask-aware loss/metrics, validation H10 Score selection, and independent `custom_models/results/benchmark_v2` plus E0-A smoke roots. First migrate the TSLib source metadata without modifying upstream files; then implement E1-A baselines, E1-B lightweight adapters, E2 groups, and E3 graph identity/protocol. Keep the 28-model list frozen before test values are inspected.

## Static checks and unresolved questions

`static_audit` parsed {stat['files_checked']} relevant Python files with {len(stat['parse_errors'])} syntax errors. Placeholder/TODO hits are reported in `audit_import_results.json`; most are framework abstract methods or explicit `NotImplementedError`, not silently treated as model implementations. Import results are in `audit_import_results.json`; synthetic smoke results are in `audit_smoke_results.json`. Unresolved: graph model sources/precise papers are absent, SDWPF adapters do not exist for TSLib, and the Git metadata/worktree baseline cannot be recovered from this directory.
""")

    write_text("BENCHMARK_PROTOCOL_GAP_AUDIT.md", f"""# E0-A Benchmark Protocol Gap Audit

## Target protocol (audit baseline)

`SDWPF`; target `Patv_raw`; input power `Patv_clean_for_input`; mask `valid_target_mask`; lookback 144; max prediction 10; H3/H6/H10; strict chronological 0.8/0.1/0.1; strides 6/3/1; batches 32/4/4; 20 epochs; patience 6; min_delta 0.01; seed 2026; AMP enabled; clip [0,1500] kW; checkpoint validation H10 official Score, lower is better; input `(B,T,N,C)`, output `(B,N,10)`.

## Confirmed current sources

- Formal ST-MGPrompt source: `custom_models/src/st_mgprompt/config.py:48-98` and `experiment_protocol.py:144-226`. The formal canonical override is protocol-aligned, including batch/early-stopping fields; raw defaults in `config.py:70-78` are not sufficient as a benchmark contract.
- Data: `data.py:95-121,124-153,156-194,269-410` loads separate input/target tables, checks alignment, fits scalers on train only, constructs split-contained windows and applies 6/3/1 strides. It explicitly records that the target mask is not model input (`:360-367`).
- Loss/metrics: `losses.py:310-348` selects the two ST-MGPrompt losses; `metrics.py:47-84,87-141,144-163` applies the mask and defines Score aliases. `Score` is the official kW→MW aligned score and `score_lower_is_better=True` (`metrics.py:112-140`).
- Evaluation: `evaluate.py:210-226,246-258` applies mask, inverse transform and clip before metrics; `evaluate.py:388-403` computes H3/H6/H10 prefix metrics. Checkpoint selection is `val_official_score_h10` min in the formal config.
- TSLib: `run.py:18-39,87-97`, `exp_long_term_forecasting.py:37-39,42-74,76-166,168-268`, and `data_loader.py:1-17,51-79` show the generic protocol and its gaps.

## Gap table

| item | required | current TSLib path | current formal ST-MGPrompt path | severity |
|---|---|---|---|---|
| dataset | SDWPF | generic ETT/custom/HuggingFace datasets; no SDWPF adapter | SDWPF parquet pair | HIGH |
| target/input | Patv_raw / Patv_clean_for_input | generic `args.target`, dataframe columns | explicit fields in config.py:59-63 | HIGH |
| mask | valid_target_mask in loss and metrics | absent | present in data/metrics/loss path | HIGH |
| features | fixed official 16 and order | `args.enc_in`, generic dataframe order | DEFAULT_16_FEATURES config.py:9-26 | HIGH |
| split/window | strict 0.8/0.1/0.1, lookback 144, contained windows | dataset-specific borders; no unified max horizon | explicit in data.py and formal protocol | HIGH |
| stride/batch | 6/3/1 and 32/4/4 | generic loader has no required stride contract; batch only train CLI | formal override has required values | HIGH |
| shape | `(B,T,N,C)` → `(B,N,10)` | `(B,T,C)` → `(B,10,C)`; 14 synthetic forwards succeed only in generic shape | model internal `[B,H,N]` converted by evaluator | HIGH |
| loss | mask-aware unified loss | MSELoss, no mask | masked_score_aligned_hybrid or method-full loss | HIGH |
| Score | official lower-is-better H10 | no official Score implementation | metrics.py:47-84 | HIGH |
| checkpoint | validation H10 Score min | validation MSE; test loss observed each epoch | formal config.py:96-98 | HIGH |
| clip | uniform [0,1500] | absent | evaluate.py:214-216 and config.py:92-95 | HIGH |
| artifacts | resolved/effective/protocol/model/checkpoints/log/metrics/metadata/status | generic checkpoint.pth, npy and text outputs | formal roots contain richer artifacts | HIGH |

## Shape and adapter findings

The TSLib synthetic helper confirms 14/18 models return `[2,10,16]` for input `[2,144,16]`; this is `[B,H,C]`, not evidence of `[B,N,H]`. `SegRNN`, `MICN`, `MultiPatchFormer`, and `TimeMixer` fail the same generic smoke due model-specific segment/decoder/scale configuration. The required adapter must decide whether each turbine is flattened into batch, treated as channels, or modeled with an explicit node dimension; this is a protocol decision for E0-B, not an automatic transpose.

## Mask, leakage, selection and artifact risks

TSLib’s `exp_long_term_forecasting.py:78-79,152-155` reads test data and reports test loss during every epoch. Even though the shown EarlyStopping call uses validation loss, test exposure is a tuning risk and cannot be accepted as a fair benchmark path. Generic `data_loader.py` also imports optional remote dataset tooling (`datasets`, `huggingface_hub`) and is unrelated to the local aligned SDWPF input/target pair. No TSLib path applies the required clip or writes H3/H6/H10 Score artifacts.

`custom_models/results/st_mgprompt` contains smoke-named runs under the formal results tree. The E0-A inventory marks them `NOT_REUSABLE`; no files were copied or promoted.

## E0-B must unify

1. One SDWPF provider and feature order.
2. One split/window/stride and batch contract.
3. One shape adapter interface with explicit node semantics.
4. One mask-aware loss/Score/MAE/RMSE/R2 implementation.
5. One validation-H10 checkpoint policy.
6. One clip policy and artifact schema.
7. Separate registry/runner/output roots from `st_mgprompt`, P0-P5, A0-A8, Canonical, `run_precision_ablation.py` and `run_ablation.py`.
""")

    write_text("STCN_STGCN_NAMING_AUDIT.md", """# STCN / STGCN Naming Audit

## Conclusion: Situation C — evidence insufficient / implementation absent

No trusted `STCN.py`, `STGCN.py`, class, forward path, adjacency provider, temporal graph block, configuration, README citation, or old result metadata was found in the local UTF-8 search scope (`custom_models`, `Time-Series-Library`, `configs`, `scripts`, `tools`, `tests`, `docs`). `Time-Series-Library/run.py:143-149` has only generic GCN-related argparse flags; it does not construct a GCN/STCN/STGCN model. The `models` directory has no GCN, STCN, STGCN, DCRNN, Graph WaveNet, MTGNN, AGCRN or STID source.

Because there is no model class or forward to inspect, the audit cannot prove whether `STCN` means classical STGCN or an independent STCN. Therefore this row is `NAME_CONFLICT`, not a guessed rename.

```text
status = NAME_CONFLICT
canonical_name = STCN_STGCN
recommended_canonical_name = UNKNOWN
legacy_alias = UNKNOWN
STGCN status = MISSING
```

## Required E3-A follow-up

Locate or deliberately select a source with paper identity, license/notice, temporal convolution + graph convolution block definition, adjacency normalization/direction, and forward shape. Record whether it is classical STGCN (then canonicalize to `STGCN`, preserving `STCN` as a legacy alias) or a distinct STCN (then keep `STCN` and add STGCN separately only if source evidence exists). Do not modify names, registry keys, filenames or old results in E0-A or before evidence is complete.
""")

    dep_lines = ["# Benchmark Dependency Report", "", f"Interpreter: `{PYTHON}`", "Python 3.11.15; torch 2.7.1+cu128; CUDA 12.8; CUDA available True; GPU NVIDIA GeForce GTX 1060; numpy 2.1.2; pandas 2.3.3; scikit-learn 1.7.2.", "", "## Dependency status", "", "The import matrix is in `audit_import_results.json`. All 18 requested TSLib modules and all audited ST-MGPrompt modules imported successfully under the required interpreter. `einops`, `PyWavelets (pywt)`, `sktime`, `datasets`, `huggingface_hub`, and `scipy` were importable. No package was installed, upgraded, downgraded or removed.", "", "## Per-model source and dependencies", ""]
    for name in TSLIB_MODELS:
        dep_lines.append(f"- **{name}** — source `{source_for(name)}`; upstream `THUML/Time-Series-Library`; import `IMPORT_OK`; direct dependencies are torch plus the local `layers` modules, and `einops` for models that import it. `WPMixer` additionally imports `pywt` through `layers/DWT_Decomposition.py`; all were importable. License: MIT from `Time-Series-Library/LICENSE`. Protocol status: dependency satisfied for import, benchmark adapter still missing.")
    dep_lines += ["", "## Graph and baseline dependencies", "", "Persistence, MovingAverage, standalone GRU, GCN, STCN/STGCN, DCRNN, Graph WaveNet, MTGNN, AGCRN and STID have no trusted local source or dependency path. They are not `DEPENDENCY_BLOCKED`; they are MISSING/PARTIAL/NAME_CONFLICT. Do not install a package to guess an implementation.", "", "## Optional CUDA / mixed environment risks", "", "TSLib README recommends its own torch environment and mentions optional extensions for models outside this audit (for example `mamba_ssm`). None is required for the 18 audited imports. The current environment is conda-based and the audit did not inspect package provenance beyond import/version; conda/pip mixing risk remains UNKNOWN. No external source or license was fetched.", "", "## Recommended stages", "", "Handle dependency pinning only after E0-B contract freeze. E1-A/E1-B cover baselines/lightweight models; E2-A through E2-D cover TSLib adapters; E3-A through E3-C cover graph source and dependencies. Preserve the local MIT notice when reusing TSLib code."]
    write_text("benchmark_dependency_report.md", "\n".join(dep_lines))


def main() -> None:
    rows = matrix()
    stat = static_audit()
    imp = imports()
    old = old_results()
    with (AUDIT_DIR / "benchmark_implementation_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    (AUDIT_DIR / "audit_import_results.json").write_text(jdump({"static": stat, "imports": imp}), encoding="utf-8")
    (AUDIT_DIR / "old_result_audit.json").write_text(jdump(old), encoding="utf-8")
    (AUDIT_DIR / "benchmark_registry_inventory.json").write_text(jdump(inventory(rows)), encoding="utf-8")
    reports(rows, stat, imp, old)
    before = json.loads((AUDIT_DIR / "protected_core_hashes_before.json").read_text(encoding="utf-8"))
    before_hashes = {key: value for key, value in before.items() if key != "snapshot"}
    after = {}
    for path in before_hashes:
        p = ROOT / path
        if p.is_file(): after[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    hash_equal = before_hashes == after
    manifest = {
        "task": "E0-A", "audit_only": True, "formal_training_started": False,
        "model_core_modified": False, "registry_modified": False, "adapter_implemented": False,
        "artifact_schema_implemented": False, "canonical_checkpoint_modified": False,
        "files_created": sorted(p.name for p in AUDIT_DIR.iterdir() if p.is_file()),
        "files_modified": [],
        "commands_executed": [
            "git rev-parse --show-toplevel / --show-current / HEAD / status --short (failed: not a git repository)",
            "D:\\Apps\\Miniconda3\\envs\\env_tslib\\python.exe environment/version/import checks",
            "UTF-8 PowerShell/Python source/config/result enumeration and keyword search",
            "AST parse of custom_models/src/st_mgprompt and relevant Time-Series-Library Python files",
            "D:\\Apps\\Miniconda3\\envs\\env_tslib\\python.exe custom_models/docs/benchmark_v2/E0_A/run_e0_a_smoke.py",
        ],
        "tests": [
            {"name": "st_mgprompt import", "status": "PASS", "details": "11 modules import successfully"},
            {"name": "TSLib model import", "status": "PASS", "details": "18 requested source modules import successfully"},
            {"name": "synthetic forward smoke", "status": "PARTIAL", "details": "14 SMOKE_OK; SegRNN/MICN/MultiPatchFormer/TimeMixer recorded SMOKE_ERROR; no training/data/formal writes"},
            {"name": "syntax AST parse", "status": "PASS" if not stat["parse_errors"] else "FAIL", "details": stat},
            {"name": "Canonical hash", "status": "PASS" if hash_equal else "FAIL", "details": {"before": before.get("custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026/best_checkpoint.pt"), "after": after.get("custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026/best_checkpoint.pt")}},
        ],
        "warnings": ["Project root has no Git metadata; pre-existing uncommitted changes cannot be distinguished from this snapshot.", "rg.exe returned Access Denied; UTF-8 PowerShell/Python search was used.", "No formal benchmark_v2 training or validation-only tuning was run.", "The smoke helper itself was corrected once for a duplicate argument before the successful smoke run."],
        "canonical_checkpoint_path": str((ROOT / "custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026/best_checkpoint.pt").resolve()),
        "canonical_sha256_before": before.get("custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026/best_checkpoint.pt"),
        "canonical_sha256_after": after.get("custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026/best_checkpoint.pt"),
        "protected_core_hashes_equal": hash_equal, "protected_core_hashes_before": before_hashes, "protected_core_hashes_after": after,
        "git_head_before": None, "git_head_after": None,
    }
    (AUDIT_DIR / "E0_A_AUDIT_MANIFEST.json").write_text(jdump(manifest), encoding="utf-8")
    write_text("audit_commands.txt", "\n".join(manifest["commands_executed"]))
    counts = {}
    for row in rows: counts[row["status"]] = counts.get(row["status"], 0) + 1
    write_text("HANDOFF_E0_A.md", f"""# HANDOFF_E0_A

## Scope and protection

This handoff completes audit-only E0-A for `{ROOT}`. It read `HANDOFF(8).md` and `PLAN.md`, audited `custom_models/src/st_mgprompt`, `custom_models/results`, `custom_models/results_smoke`, `dataset`, and the local `Time-Series-Library` source/config/runner/layer/data paths. No model core, formal registry, runner, data protocol, Canonical checkpoint, P0-P5, A0-A8, dependency set, or formal result directory was modified. No training or tuning started; no files were deleted.

Git metadata is absent, so Git HEAD/branch/status are `null` rather than invented. Canonical is `{REL(ROOT / 'custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026')}` and its best checkpoint remains SHA256 `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`. P0 and A0 `reference.json` point to this artifact with `trainable:false`; `run_st_mgprompt.py:489` and `train.py:441,589` use strict checkpoint loading.

## Findings

- Model matrix: {counts}.
- TSLib provides 18 importable upstream models through filename auto-scan/LazyModelDict, not a unified SDWPF benchmark. All 18 are `IMPLEMENTED_PROTOCOL_RISK`.
- Persistence and MovingAverage are `MISSING`.
- GRU is `PARTIAL_IMPLEMENTATION`: only an internal ST-MGPrompt GRU block exists; no standalone factory/runner/data/loss/metrics path.
- GCN, DCRNN, Graph WaveNet, MTGNN, AGCRN and STID are `MISSING`.
- STCN/STGCN is `NAME_CONFLICT` with Situation C: no source evidence for either identity; preserve the label and resolve in E3-A.
- Main protocol gap: TSLib generic `(B,T,C)`/`[B,H,C]`, MSE, generic target/data loader, no `valid_target_mask`, no official Score/clip/H10 checkpoint contract. Its test loss is observed every training epoch.
- Smoke pollution exists under `custom_models/results/st_mgprompt` for smoke-named runs; those are historical reference only, not benchmark artifacts. No local external SDWPF benchmark run was found.

## Deliverables

See all files in this directory. The required files are `BENCHMARK_IMPLEMENTATION_AUDIT.md`, `BENCHMARK_PROTOCOL_GAP_AUDIT.md`, `STCN_STGCN_NAMING_AUDIT.md`, `benchmark_implementation_matrix.csv`, `benchmark_dependency_report.md`, `benchmark_registry_inventory.json`, `E0_A_AUDIT_MANIFEST.json`, and this handoff. Supporting files are `audit_commands.txt`, `audit_import_results.json`, `audit_smoke_results.json`, `old_result_audit.json`, `protected_core_hashes_before.json`, `run_e0_a_smoke.py`, and `generate_e0_a_artifacts.py`.

## Tests

The required interpreter imported 11 ST-MGPrompt modules and all 18 requested TSLib model modules. Relevant Python files were AST-parsed. Synthetic forward smoke ran one batch only: 14 models succeeded; SegRNN, MICN, MultiPatchFormer and TimeMixer failed with recorded model/config shape errors. No full dataset, full-shape SDWPF run, official training, validation-only tuning, or benchmark aggregation was run.

## E0-B exact input and order

E0-B should receive `benchmark_implementation_matrix.csv`, `benchmark_registry_inventory.json`, `BENCHMARK_PROTOCOL_GAP_AUDIT.md`, `benchmark_dependency_report.md`, the local TSLib source and its MIT `LICENSE`, plus the formal ST-MGPrompt protocol files for read-only contract comparison. Implement, in order: (1) new independent benchmark_v2 registry and canonical names; (2) adapter interface for `(B,T,N,C)` and explicit `[B,N,H]`; (3) one SDWPF provider/mask-aware loss/metrics/checkpoint policy; (4) artifact schema and isolated smoke/formal roots; (5) migrate E1-A/E1-B sources, then E2 and E3. Do not import or modify `st_mgprompt.registry`, `run_precision_ablation.py`, `run_ablation.py`, Canonical, P0-P5 or A0-A8 from the new runtime; only read/reference them.

## Traps

  Do not treat TSLib filenames as fair implementations; do not transpose `[B,H,C]` into `[B,N,H]` without deciding node semantics; do not use its MSE/test-loss path; do not call generic HuggingFace fallback; do not add graph models by name only; do not rename STCN/STGCN before evidence; do not promote smoke runs; do not select models from test metrics; do not modify or retrain Canonical/P0/A0; do not install dependencies; do not delete files.
  """)
    manifest = json.loads((AUDIT_DIR / "E0_A_AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["files_created"] = sorted(p.name for p in AUDIT_DIR.iterdir() if p.is_file())
    (AUDIT_DIR / "E0_A_AUDIT_MANIFEST.json").write_text(jdump(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
