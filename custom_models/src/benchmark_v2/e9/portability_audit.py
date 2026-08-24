from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT
from .io_utils import file_identity

LOSS_SOURCE = PROJECT_ROOT / "custom_models" / "src" / "st_mgprompt" / "losses.py"
ADAPTER_SOURCE = PROJECT_ROOT / "custom_models" / "src" / "benchmark_v2" / "losses.py"

MODEL_INTERNAL_TOKENS = {
    "fine", "coarse", "vadsp", "prompt", "cross_gate", "hidden_state",
    "attention", "graph_embedding", "model_aux", "macro_prompt", "st_prompt",
}


def classify_portability(
    *, requires_train_state: bool, model_internal_dependencies: list[str] | tuple[str, ...],
    semantics_preserved_without_internal: bool = False, portable_subset_name: str | None = None,
) -> dict[str, Any]:
    internal = sorted(set(model_internal_dependencies))
    if not internal:
        classification = "PORTABLE_WITH_TRAIN_STATE" if requires_train_state else "FULLY_PORTABLE"
        return {
            "classification": classification, "e9_b_allowed": True,
            "FULL_MS_MG_DWU_TRANSFER_READY": True, "PORTABLE_SUBSET_TRANSFER_READY": False,
            "portable_subset_required": False, "portable_subset_name": None,
        }
    if not semantics_preserved_without_internal:
        return {
            "classification": "NOT_PORTABLE", "e9_b_allowed": False,
            "FULL_MS_MG_DWU_TRANSFER_READY": False, "PORTABLE_SUBSET_TRANSFER_READY": False,
            "portable_subset_required": False, "portable_subset_name": None,
        }
    if not portable_subset_name or portable_subset_name.lower() in {"msmg_dwu", "ms-mg-dwu", "msmg_dwu_loss"}:
        raise ValueError("A model-independent subset must use a new explicit loss name.")
    return {
        "classification": "MODEL_INTERNAL_DEPENDENT", "e9_b_allowed": True,
        "FULL_MS_MG_DWU_TRANSFER_READY": False, "PORTABLE_SUBSET_TRANSFER_READY": True,
        "portable_subset_required": True, "portable_subset_name": portable_subset_name,
    }


def _forward_signature() -> tuple[list[str], set[str]]:
    tree = ast.parse(LOSS_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MSMGDWULoss":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    names = [arg.arg for arg in item.args.args]
                    used = {part.id for part in ast.walk(item) if isinstance(part, ast.Name)}
                    return names, used
    raise RuntimeError("MSMGDWULoss.forward was not found in the audited source.")


def portability_audit() -> dict[str, Any]:
    arguments, used = _forward_signature()
    internal_used = sorted(MODEL_INTERNAL_TOKENS & used)
    expected = ["self", "pred", "target", "mask"]
    dependency_rows = [
        ("prediction", True, "public model output; gradient required"),
        ("target", True, "normalized target from formal batch"),
        ("mask", True, "valid_target_mask from formal batch"),
        ("site/node identity", True, "semantic N axis; weights are indexed by the frozen node order"),
        ("horizon identity", True, "fixed public prefixes H3/H6/H10"),
        ("normalization state", True, "loss operates in train-fitted normalized target space"),
        ("training-only fitted statistics", True, "online EMA/error state updates only while loss.training is true"),
        ("Fine representation", False, "not referenced"),
        ("Coarse representation", False, "not referenced"),
        ("granularity / VADSP state", False, "granularity means public horizon prefix, not VADSP representation"),
        ("Macro Prompt", False, "not referenced"),
        ("ST Prompt", False, "not referenced"),
        ("Cross gate", False, "not referenced"),
        ("graph embedding", False, "not referenced"),
        ("hidden state", False, "not referenced"),
        ("model-specific auxiliary tensor", False, "not accepted by forward"),
        ("ST-MGPrompt proprietary object", False, "loss module is independent of the model instance"),
    ]
    signature_valid = arguments == expected and not internal_used
    decision = classify_portability(
        requires_train_state=True,
        model_internal_dependencies=internal_used if arguments == expected else ["invalid_forward_contract"],
        semantics_preserved_without_internal=False,
    )
    return {
        "schema_version": "e9_portability_audit_v1",
        "audit_stage": "E9-A",
        "source": file_identity(LOSS_SOURCE),
        "adapter_source": file_identity(ADAPTER_SOURCE),
        "source_symbol": "st_mgprompt.losses.MSMGDWULoss",
        "forward_signature": arguments,
        "forward_signature_valid": signature_valid,
        "dependency_closure": [
            {"dependency": name, "required": required, "finding": finding}
            for name, required, finding in dependency_rows
        ],
        "model_internal_tokens_used_in_forward": internal_used,
        "train_state": {
            "fields": [
                "ema_granularity_loss", "initial_granularity_loss", "initial_granularity_fitted",
                "ema_node_loss", "node_weight",
            ],
            "fit_split": "train batches only",
            "update_guard": "self.training",
            "validation_test_frozen": True,
            "model_output_dependent_state": True,
            "model_independent_algorithm": True,
            "shared_values_across_models": False,
            "reason": "EMA values depend on each model's training predictions; the frozen algorithm/profile is shared.",
        },
        "classification": decision["classification"],
        "classification_reason": (
            "The exact loss consumes only public prediction/target/mask with fixed site and horizon axes. "
            "Its additional state is training-only, model-output-dependent EMA state governed by one model-independent algorithm and frozen in validation/test."
        ),
        "FULL_MS_MG_DWU_TRANSFER_READY": bool(signature_valid and decision["FULL_MS_MG_DWU_TRANSFER_READY"]),
        "PORTABLE_SUBSET_TRANSFER_READY": decision["PORTABLE_SUBSET_TRANSFER_READY"],
        "portable_subset_required": decision["portable_subset_required"],
        "e9_b_allowed": bool(signature_valid and decision["e9_b_allowed"]),
        "e5_consumed": False,
    }
