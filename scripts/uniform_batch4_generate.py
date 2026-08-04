from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
sys.path.insert(0, str(SRC_ROOT))

from benchmark_v2.experiments.e5_common_loss.legacy_scope29_contract import (  # noqa: E402
    NONTRAINABLE_MODELS,
    TRAINABLE_MODELS,
)
from benchmark_v2.experiments.e5_common_loss.variant_manifest import (  # noqa: E402
    build_run_id_map as build_e5_run_id_map,
    extract_base_run_ids,
)
from benchmark_v2.hardware_preflight import (  # noqa: E402
    preflight_artifact_path,
)
from benchmark_v2.protocol import load_protocol  # noqa: E402
from benchmark_v2.registry import load_registry  # noqa: E402
from benchmark_v2.training_profiles import (  # noqa: E402
    UNIFORM_BATCH4_PROFILE_ID,
    load_training_profile,
    stable_hash,
)


DOC_ROOT = (
    PROJECT_ROOT
    / "custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL"
)
PREFLIGHT_ROOT = (
    PROJECT_ROOT
    / "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
)
PYTHON_WINDOWS = r"D:\Apps\Miniconda3\envs\env_tslib\python.exe"
PYTHON_LINUX = "${PYTHON:-python}"
PROFILE_ID = UNIFORM_BATCH4_PROFILE_ID
KNOWN_OOM6 = (
    "segrnn",
    "transformer",
    "patchtst",
    "frets",
    "msgnet",
    "timefilter",
)
TRAINABLE = tuple(TRAINABLE_MODELS)
NONTRAINABLE = tuple(NONTRAINABLE_MODELS)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def bs4_run_id(base_run_id: str) -> str:
    if not base_run_id.endswith("_seed2026"):
        raise ValueError(base_run_id)
    return base_run_id[: -len("_seed2026")] + "_bs4_seed2026"


def output_group(model_id: str) -> str:
    registry = load_registry().get(model_id)
    stage = str(registry.planned_stage).lower().replace("-", "_")
    return {
        "e1_a": "basic_lightweight_seed2026",
        "e1_b": "basic_lightweight_seed2026",
        "e2_a": "e2_a_seed2026",
        "e2_b": "e2_b_seed2026",
        "e2_c": "e2_c_seed2026",
        "e2_d": "e2_d_seed2026",
        "e3_b": "e3_b_seed2026",
        "e3_c": "e3_c_seed2026",
    }.get(stage, "basic_lightweight_seed2026")


def build_run_map() -> dict[str, Any]:
    profile = load_training_profile(PROFILE_ID)
    base = extract_base_run_ids()
    runs = []
    for model_id in (*NONTRAINABLE, *TRAINABLE):
        mode = "EVALUATE_ONLY" if model_id in NONTRAINABLE else "TRAIN"
        group = output_group(model_id)
        runs.append(
            {
                "model_id": model_id,
                "experiment_type": "original_benchmark",
                "base_run_id": base[model_id]["base_run_id"],
                "new_run_id": bs4_run_id(base[model_id]["base_run_id"]),
                "batch_profile_id": PROFILE_ID,
                "output_root": (
                    f"custom_models/results/benchmark_v2_uniform_bs4/{group}"
                ),
                "mode": mode,
                "loss_id": "NOT_APPLICABLE" if mode == "EVALUATE_ONLY" else "masked_mse",
            }
        )
    e5_map = build_e5_run_id_map(PROFILE_ID)["runs"]
    for model_id in (*NONTRAINABLE, *TRAINABLE):
        runs.append(
            {
                "model_id": model_id,
                "experiment_type": "e5_common_loss",
                "base_run_id": base[model_id]["base_run_id"],
                "new_run_id": e5_map[model_id]["e5_run_id"],
                "batch_profile_id": PROFILE_ID,
                "output_root": (
                    "custom_models/results/benchmark_v2_uniform_bs4/"
                    "common_loss_architecture_seed2026"
                ),
                "mode": e5_map[model_id]["mode"],
                "loss_id": (
                    "masked_score_aligned_hybrid"
                    if model_id in TRAINABLE
                    else "NOT_APPLICABLE"
                ),
            }
        )
    runs.extend(
        [
            {
                "model_id": "st_mgprompt_full",
                "experiment_type": "st_mgprompt_full",
                "base_run_id": "full_fixed_dual_keep_msmgdwu_seed2026",
                "new_run_id": "full_fixed_dual_keep_msmgdwu_bs4_seed2026",
                "batch_profile_id": PROFILE_ID,
                "output_root": "custom_models/results/st_mgprompt_uniform_bs4",
                "mode": "TRAIN",
                "loss_id": "msmg_dwu_loss",
            },
            {
                "model_id": "st_mgprompt_a8",
                "experiment_type": "st_mgprompt_a8",
                "base_run_id": "component_ablation_fixed_dual_seed2026/A8",
                "new_run_id": "component_ablation_a8_bs4_seed2026",
                "batch_profile_id": PROFILE_ID,
                "output_root": "custom_models/results/st_mgprompt_uniform_bs4",
                "mode": "TRAIN",
                "loss_id": "masked_score_aligned_hybrid",
                "reference_id": "STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference",
            },
        ]
    )
    return {
        "schema_version": "uniform_batch4_run_id_map_v1",
        "profile_id": PROFILE_ID,
        "profile_hash": profile.profile_hash,
        "runs": runs,
    }


def pwsh_header() -> str:
    return f"""$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$Python = "{PYTHON_WINDOWS}"
$Profile = "{PROFILE_ID}"
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "custom_models\\src"
"""


def bash_header() -> str:
    return f"""#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN={PYTHON_LINUX}
PROFILE="{PROFILE_ID}"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
"""


def windows_precheck() -> str:
    return pwsh_header() + r"""
$active = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -match "^(python|pythonw|powershell|pwsh|cmd)(\.exe)?$") -and
  ($_.CommandLine -match "(run_benchmark\.py\s+(train|evaluate-only)|hardware[-_]preflight|run_st_mgprompt\.py.*--full|UNIFORM_BATCH4_RUN_)")
}
if ($active) { $active | Format-List; throw "BLOCKED_ACTIVE_FORMAL_RUN" }
$activeStatus = @()
Get-ChildItem -LiteralPath "custom_models" -Filter "run_status.json" -File -Recurse | ForEach-Object {
  try {
    $value = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($value.status -in @("RUNNING","STARTED","TRAINING","EVALUATING","PREFLIGHT_RUNNING") -and
        $value.formal_training -ne $false -and $value.run_mode -ne "smoke") {
      $activeStatus += $_.FullName
    }
  } catch {}
}
if ($activeStatus.Count -gt 0) { $activeStatus; throw "BLOCKED_ACTIVE_FORMAL_RUN" }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" protocol-check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
"""


def linux_precheck() -> str:
    return bash_header() + r"""
if pgrep -af 'run_benchmark.py (train|evaluate-only)|hardware[-_]preflight|run_st_mgprompt.py.*--full|UNIFORM_BATCH4_RUN_' >/tmp/uniform_batch4_active.txt; then
  cat /tmp/uniform_batch4_active.txt
  exit 20
fi
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py protocol-check
"""


def windows_preflight(models: tuple[str, ...], suite: str) -> str:
    lines = [pwsh_header(), "$Failed = 0"]
    for model in models:
        lines.extend(
            [
                f'& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "{model}" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"',
                "if ($LASTEXITCODE -ne 0) { $Failed = 1 }",
            ]
        )
    lines.append(
        f'& $Python "scripts/uniform_batch4_generate.py" collect-preflight --suite "{suite}"'
    )
    lines.append("if ($Failed -ne 0) { exit 3 }")
    return "\n".join(lines)


def linux_preflight(models: tuple[str, ...], suite: str) -> str:
    lines = [bash_header(), "failed=0"]
    for model in models:
        lines.extend(
            [
                f'"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model "{model}" --training-profile "$PROFILE" --preflight-root custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight || failed=1',
            ]
        )
    lines.append(
        f'"$PYTHON_BIN" scripts/uniform_batch4_generate.py collect-preflight --suite "{suite}"'
    )
    lines.append('exit "$failed"')
    return "\n".join(lines)


def original_commands(windows: bool) -> str:
    run_map = build_run_map()["runs"]
    rows = [r for r in run_map if r["experiment_type"] == "original_benchmark"]
    lines = [pwsh_header() if windows else bash_header()]
    for row in rows:
        command = "evaluate-only" if row["mode"] == "EVALUATE_ONLY" else "train"
        if windows:
            log = (
                f'custom_models/logs/uniform_bs4/original/{row["model_id"]}.log'
            )
            lines.extend(
                [
                    f'New-Item -ItemType Directory -Force -Path (Split-Path -Parent "{log}") | Out-Null',
                    f'& $Python "custom_models/src/benchmark_v2/run_benchmark.py" {command} --model "{row["model_id"]}" --training-profile $Profile --output-root "{row["output_root"]}" --run-id "{row["new_run_id"]}" --device cuda *>&1 | Tee-Object -FilePath "{log}"',
                    "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
                ]
            )
        else:
            log = f'custom_models/logs/uniform_bs4/original/{row["model_id"]}.log'
            lines.extend(
                [
                    f'mkdir -p "$(dirname "{log}")"',
                    f'"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py {command} --model "{row["model_id"]}" --training-profile "$PROFILE" --output-root "{row["output_root"]}" --run-id "{row["new_run_id"]}" --device cuda 2>&1 | tee "{log}"',
                    'code=${PIPESTATUS[0]}; if [ "$code" -ne 0 ]; then exit "$code"; fi',
                ]
            )
    return "\n".join(lines)


def st_command(variant: str, windows: bool) -> str:
    is_a8 = variant == "A8"
    run_id = (
        "component_ablation_a8_bs4_seed2026"
        if is_a8
        else "full_fixed_dual_keep_msmgdwu_bs4_seed2026"
    )
    variant_flag = "--component-ablation A8" if is_a8 else "--canonical-full"
    log_name = "a8" if is_a8 else "full"
    if windows:
        return pwsh_header() + f"""
New-Item -ItemType Directory -Force -Path "custom_models/logs/uniform_bs4" | Out-Null
& $Python "custom_models/src/st_mgprompt/run_st_mgprompt.py" --full {variant_flag} --training-profile $Profile --run-id "{run_id}" --output-root "custom_models/results/st_mgprompt_uniform_bs4" --windows-safe-mode --amp *>&1 | Tee-Object -FilePath "custom_models/logs/uniform_bs4/{log_name}.log"
exit $LASTEXITCODE
"""
    return bash_header() + f"""
mkdir -p custom_models/logs/uniform_bs4
"$PYTHON_BIN" custom_models/src/st_mgprompt/run_st_mgprompt.py --full {variant_flag} --training-profile "$PROFILE" --run-id "{run_id}" --output-root "custom_models/results/st_mgprompt_uniform_bs4" --amp 2>&1 | tee "custom_models/logs/uniform_bs4/{log_name}.log"
exit ${{PIPESTATUS[0]}}
"""


def e5_commands(windows: bool) -> str:
    rows = [
        row
        for row in build_run_map()["runs"]
        if row["experiment_type"] == "e5_common_loss"
    ]
    lines = [pwsh_header() if windows else bash_header()]
    for row in rows:
        command = "evaluate-only" if row["mode"] == "EVALUATE_ONLY" else "train"
        extra = "--experiment-profile e5_common_loss_v1"
        if windows:
            log = f'custom_models/logs/uniform_bs4/e5/{row["model_id"]}.log'
            lines.extend(
                [
                    f'New-Item -ItemType Directory -Force -Path (Split-Path -Parent "{log}") | Out-Null',
                    f'& $Python "custom_models/src/benchmark_v2/run_benchmark.py" {command} --model "{row["model_id"]}" {extra} --training-profile $Profile --output-root "{row["output_root"]}" --run-id "{row["new_run_id"]}" --device cuda *>&1 | Tee-Object -FilePath "{log}"',
                    "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
                ]
            )
        else:
            log = f'custom_models/logs/uniform_bs4/e5/{row["model_id"]}.log'
            lines.extend(
                [
                    f'mkdir -p "$(dirname "{log}")"',
                    f'"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py {command} --model "{row["model_id"]}" {extra} --training-profile "$PROFILE" --output-root "{row["output_root"]}" --run-id "{row["new_run_id"]}" --device cuda 2>&1 | tee "{log}"',
                    'code=${PIPESTATUS[0]}; if [ "$code" -ne 0 ]; then exit "$code"; fi',
                ]
            )
    return "\n".join(lines)


def autoshutdown(command_name: str) -> str:
    return bash_header() + f"""
set +e
bash "$PROJECT_ROOT/custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/{command_name}"
code=$?
sync
/usr/bin/shutdown -h now
exit $code
"""


def readiness_script(windows: bool) -> str:
    if windows:
        return pwsh_header() + r"""
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" e5-readiness --training-profile $Profile --output-root "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026" --report-path "custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/E5_BATCH4_READINESS.json"
exit $LASTEXITCODE
"""
    return bash_header() + r"""
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py e5-readiness --training-profile "$PROFILE" --output-root custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026 --report-path custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/E5_BATCH4_READINESS.json
"""


def generate_scripts() -> None:
    def pwsh_gate(arguments: str) -> str:
        return (
            pwsh_header()
            + f'\n& $Python "scripts/uniform_batch4_machine_gate.py" {arguments}\n'
            + "exit $LASTEXITCODE\n"
        )

    def bash_gate(arguments: str) -> str:
        return (
            bash_header()
            + f'\n"$PYTHON_BIN" scripts/uniform_batch4_machine_gate.py {arguments}\n'
            + "exit $?\n"
        )

    scripts = {
        "UNIFORM_BATCH4_PRECHECK_WINDOWS.ps1": (
            windows_precheck()
            + '\n& $Python "scripts/uniform_batch4_machine_gate.py" precheck\n'
            + "exit $LASTEXITCODE\n"
        ),
        "UNIFORM_BATCH4_PREFLIGHT_ORIGINAL_ALL26_WINDOWS.ps1": pwsh_gate(
            'preflight-inventory --suite original --python "$Python"'
        ),
        "UNIFORM_BATCH4_PREFLIGHT_E5_ALL26_WINDOWS.ps1": pwsh_gate(
            'preflight-inventory --suite e5 --python "$Python"'
        ),
        "UNIFORM_BATCH4_PREFLIGHT_FULL_WINDOWS.ps1": pwsh_gate(
            'preflight-st --suite full --python "$Python"'
        ),
        "UNIFORM_BATCH4_PREFLIGHT_A8_WINDOWS.ps1": pwsh_gate(
            'preflight-st --suite a8 --python "$Python"'
        ),
        "UNIFORM_BATCH4_RUN_ORIGINAL_28_WINDOWS.ps1": pwsh_gate(
            'formal-exec --suite original --python "$Python"'
        ),
        "UNIFORM_BATCH4_RUN_FULL_WINDOWS.ps1": pwsh_gate(
            'formal-exec --suite full --python "$Python"'
        ),
        "UNIFORM_BATCH4_RUN_A8_WINDOWS.ps1": pwsh_gate(
            'formal-exec --suite a8 --python "$Python"'
        ),
        "UNIFORM_BATCH4_RUN_E5_CORE_28_WINDOWS.ps1": pwsh_gate(
            'formal-exec --suite e5_core --python "$Python"'
        ),
        "UNIFORM_BATCH4_FINALIZE_E5_WINDOWS.ps1": pwsh_gate(
            'finalize-e5 --python "$Python"'
        ),
        "UNIFORM_BATCH4_RUN_E5_WINDOWS.ps1": (
            pwsh_header()
            + '\nWrite-Warning "Deprecated: delegating to E5_CORE_28. Finalization is separate."\n'
            + '& (Join-Path $PSScriptRoot "UNIFORM_BATCH4_RUN_E5_CORE_28_WINDOWS.ps1")\n'
            + "exit $LASTEXITCODE\n"
        ),
        "UNIFORM_BATCH4_READINESS_WINDOWS.ps1": (
            pwsh_header()
            + '\nWrite-Error "Deprecated partial readiness entry. Use UNIFORM_BATCH4_FINALIZE_E5_WINDOWS.ps1 after base28 and batch4 A8 complete."\n'
            + "exit 64\n"
        ),
        "UNIFORM_BATCH4_PRECHECK_LINUX.sh": (
            linux_precheck()
            + '\n"$PYTHON_BIN" scripts/uniform_batch4_machine_gate.py precheck\n'
            + "exit $?\n"
        ),
        "UNIFORM_BATCH4_PREFLIGHT_ORIGINAL_ALL26_LINUX.sh": bash_gate(
            'preflight-inventory --suite original --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_PREFLIGHT_E5_ALL26_LINUX.sh": bash_gate(
            'preflight-inventory --suite e5 --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_PREFLIGHT_FULL_LINUX.sh": bash_gate(
            'preflight-st --suite full --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_PREFLIGHT_A8_LINUX.sh": bash_gate(
            'preflight-st --suite a8 --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_RUN_ORIGINAL_28_LINUX.sh": bash_gate(
            'formal-exec --suite original --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_RUN_FULL_LINUX.sh": bash_gate(
            'formal-exec --suite full --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_RUN_A8_LINUX.sh": bash_gate(
            'formal-exec --suite a8 --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_RUN_E5_CORE_28_LINUX.sh": bash_gate(
            'formal-exec --suite e5_core --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_FINALIZE_E5_LINUX.sh": bash_gate(
            'finalize-e5 --python "$PYTHON_BIN"'
        ),
        "UNIFORM_BATCH4_RUN_E5_LINUX.sh": (
            bash_header()
            + '\necho "Deprecated: delegating to E5_CORE_28. Finalization is separate." >&2\n'
            + 'bash "$PROJECT_ROOT/custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_E5_CORE_28_LINUX.sh"\n'
            + "exit $?\n"
        ),
        "UNIFORM_BATCH4_READINESS_LINUX.sh": (
            bash_header()
            + '\necho "Deprecated partial readiness entry. Use UNIFORM_BATCH4_FINALIZE_E5_LINUX.sh after base28 and batch4 A8 complete." >&2\n'
            + "exit 64\n"
        ),
    }
    for base in ("ORIGINAL_28", "FULL", "A8", "E5_CORE_28"):
        scripts[f"UNIFORM_BATCH4_RUN_{base}_LINUX_AUTOSHUTDOWN.sh"] = autoshutdown(
            f"UNIFORM_BATCH4_RUN_{base}_LINUX.sh"
        )
    scripts["UNIFORM_BATCH4_RUN_E5_LINUX_AUTOSHUTDOWN.sh"] = autoshutdown(
        "UNIFORM_BATCH4_RUN_E5_LINUX.sh"
    )
    for name, content in scripts.items():
        path = DOC_ROOT / name
        if path.suffix == ".sh":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((content.rstrip() + "\n").encode("utf-8"))
            path.chmod(path.stat().st_mode | 0o111)
        else:
            write_text(path, content)


def placeholder(models: tuple[str, ...], suite: str) -> dict[str, Any]:
    profile = load_training_profile(PROFILE_ID)
    return {
        "schema_version": "uniform_batch4_preflight_inventory_v1",
        "suite": suite,
        "status": "NOT_RUN",
        **profile.identity(),
        "models": [
            {
                "model_id": model,
                "status": "NOT_RUN",
                "forward_completed": False,
                "backward_completed": False,
                "peak_allocated_memory": None,
                "peak_reserved_memory": None,
                "failure_stage": None,
            }
            for model in models
        ],
        "counts": {
            "PASS": 0,
            "FAIL_OOM": 0,
            "FAIL_NON_OOM": 0,
            "NOT_RUN": len(models),
        },
    }


def collect_preflight(suite: str) -> dict[str, Any]:
    models = KNOWN_OOM6 if suite == "known_oom6" else TRAINABLE
    rows = []
    for model in models:
        path = preflight_artifact_path(
            model,
            root=PREFLIGHT_ROOT,
            training_profile=PROFILE_ID,
        )
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["artifact_path"] = str(path)
            rows.append(payload)
        else:
            rows.append(
                {
                    "model_id": model,
                    "status": "NOT_RUN",
                    "forward_completed": False,
                    "backward_completed": False,
                    "peak_allocated_memory": None,
                    "peak_reserved_memory": None,
                    "failure_stage": None,
                    "artifact_path": str(path),
                }
            )
    counts = {
        status: sum(row.get("status") == status for row in rows)
        for status in ("PASS", "FAIL_OOM", "FAIL_NON_OOM", "NOT_RUN")
    }
    if counts["FAIL_NON_OOM"]:
        status = "BLOCKED_BATCH4_NON_OOM"
    elif counts["FAIL_OOM"]:
        status = "BLOCKED_BATCH4_STILL_OOM"
    elif counts["NOT_RUN"]:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    profile = load_training_profile(PROFILE_ID)
    report = {
        "schema_version": "uniform_batch4_preflight_inventory_v1",
        "suite": suite,
        "status": status,
        **profile.identity(),
        "models": rows,
        "counts": counts,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    target = (
        "BATCH4_KNOWN_OOM6_PREFLIGHT_RESULTS.json"
        if suite == "known_oom6"
        else "BATCH4_ALL26_PREFLIGHT_RESULTS.json"
    )
    write_json(DOC_ROOT / target, report)
    return report


def generate_docs() -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    profile = load_training_profile(PROFILE_ID)
    combined = {
        "schema_version": "uniform_batch4_protocol_v1",
        "protocol_id": "uniform_batch4_protocol_v1",
        "base_benchmark_protocol_hash": protocol.protocol_hash,
        **profile.identity(),
        "optimizer_step_policy": (
            "No gradient accumulation. Optimizer steps per epoch increase "
            "relative to batch=32."
        ),
        "scope": {
            "original_trainable": 26,
            "original_evaluate_only": 2,
            "st_mgprompt_full": 1,
            "st_mgprompt_a8": 1,
            "e5_trainable": 26,
            "e5_evaluate_only": 2,
        },
    }
    combined_hash = stable_hash(combined)
    write_json(DOC_ROOT / "UNIFORM_BATCH4_PROTOCOL.json", combined)
    write_json(
        DOC_ROOT / "UNIFORM_BATCH4_PROTOCOL_HASH.json",
        {"protocol_hash": combined_hash, "hash_algorithm": "sha256_canonical_json"},
    )
    write_json(DOC_ROOT / "UNIFORM_BATCH4_PROFILE.json", profile.payload)
    write_json(
        DOC_ROOT / "UNIFORM_BATCH4_PROFILE_HASH.json",
        {
            "profile_id": profile.profile_id,
            "training_batch_profile_hash": profile.profile_hash,
        },
    )
    write_json(DOC_ROOT / "UNIFORM_BATCH4_RUN_ID_MAP.json", build_run_map())
    write_json(
        DOC_ROOT / "BATCH4_KNOWN_OOM6_PREFLIGHT_RESULTS.json",
        placeholder(KNOWN_OOM6, "known_oom6"),
    )
    write_json(
        DOC_ROOT / "BATCH4_ALL26_PREFLIGHT_RESULTS.json",
        placeholder(TRAINABLE, "all26"),
    )
    for name, object_id in (
        ("BATCH4_ST_MGPROMPT_FULL_PREFLIGHT.json", "st_mgprompt_full"),
        ("BATCH4_A8_PREFLIGHT.json", "st_mgprompt_a8"),
    ):
        write_json(
            DOC_ROOT / name,
            {
                "schema_version": "st_mgprompt_batch4_preflight_v1",
                "object_id": object_id,
                "status": "NOT_RUN",
                **profile.identity(),
            },
        )
    write_text(
        DOC_ROOT / "UNIFORM_BATCH4_DESIGN.md",
        f"""# Uniform Batch=4 Design

`{PROFILE_ID}` is the only new training batch profile. It overlays the frozen
Benchmark Protocol v1 (`{protocol.protocol_hash}`) without changing that file
or its logical hash. Train/validation/test batches are 4/4/4, AMP is enabled,
gradient accumulation is 1, and effective train batch is 4.

The overlay changes DataLoader batch formation, artifact identity, run-id,
output namespace, and preflight identity only. It does not enter model
construction. Batch fallback, per-model batches, activation checkpointing,
offload, model parallelism, pruning, quantization, and capacity changes are
forbidden. Batch=4 creates more optimizer updates per epoch than batch=32.
""",
    )
    write_text(
        DOC_ROOT / "BATCH4_MEMORY_FEASIBILITY_REPORT.md",
        """# Batch=4 Memory Feasibility

Exact feasibility is measured at B=4, T=144, N=134, C=16, H=10 with AMP.
Each benchmark model runs in an independent Python process. The known OOM six
inventory is the first gate; all 26, Full, and A8 are permitted only after it
passes. Results are machine-specific and must not be inferred from batch=32.
""",
    )
    plans = {
        "ORIGINAL_28_BATCH4_RETRAIN_PLAN.md": "26 trainable models retain masked MSE; Persistence and MovingAverage remain evaluate-only.",
        "ST_MGPROMPT_FULL_BATCH4_RETRAIN_PLAN.md": "Retrain canonical Full architecture and method_full/MS-MG-DWU loss; mark output BATCH4_CANONICAL_CANDIDATE.",
        "A8_BATCH4_RETRAIN_PLAN.md": "Retrain A8 (w/o MS-MG-DWU) with masked_score_aligned_hybrid.",
        "E5_BATCH4_RETRAIN_PLAN.md": "Retrain 26 architectures with masked_score_aligned_hybrid, evaluate two deterministic baselines, then require the new batch4 A8 reference.",
    }
    for name, body in plans.items():
        write_text(
            DOC_ROOT / name,
            f"# {name.removesuffix('.md').replace('_', ' ')}\n\n{body}\n\nNo formal run is started by this implementation task.",
        )
    write_text(
        DOC_ROOT / "UNIFORM_BATCH4_RESULT_SCHEMA.md",
        """# Uniform Batch=4 Result Schema

Every artifact and checkpoint must carry
`base_benchmark_protocol_hash`, `training_batch_profile_id`,
`training_batch_profile_hash`, train/val/test batch sizes,
`gradient_accumulation_steps`, and `effective_train_batch_size`.
Formal trainable runs also carry loss identity, model config/source closure
identity, graph/node/timestamp identity, and matching exact preflight identity.
""",
    )
    write_text(
        DOC_ROOT / "UNIFORM_BATCH4_AGGREGATION_POLICY.md",
        """# Uniform Batch=4 Aggregation Policy

E4 may read only batch4 original benchmark artifacts and batch4 Full. E5 may
read only batch4 common-loss artifacts and the new batch4 A8 reference.
Readiness rejects any id/hash/batch/loss/config/source mismatch as
`BLOCKED_MIXED_BATCH_PROFILE`; aggregation must not emit a final table.
""",
    )
    write_text(
        DOC_ROOT / "UNIFORM_BATCH4_RUNBOOK.md",
        """# Uniform Batch=4 Runbook

1. Run the platform PRECHECK script.
2. Run KNOWN_OOM_6 preflight. Stop on any OOM or non-OOM failure.
3. Run ALL_26 preflight. It records every model and does not fail-fast.
4. Run Full and A8 exact preflights and verify strict reload.
5. Only after all four gates pass: original 28, Full, A8, E5, E4 aggregation,
   then E5 aggregation.

Formal suite scripts run one Python process at a time, stop at the first failed
formal run, preserve logs/exit codes, and rely on fail-closed non-overwrite run
directories. Linux auto-shutdown wrappers always capture the suite exit code,
sync, call `/usr/bin/shutdown -h now`, and return the captured code.
""",
    )
    write_text(
        DOC_ROOT / "IMPLEMENTATION_REPORT.md",
        """# Implementation Report

Implemented a versioned batch4 overlay, CLI/DataLoader injection, batch-aware
preflight/artifact/checkpoint identities, mixed-batch readiness rejection,
ST-MGPrompt Full/A8 batch identity, tests, protected snapshots, and platform
run scripts. No formal training or aggregation is executed by generation.
""",
    )
    write_text(
        DOC_ROOT / "HANDOFF_UNIFORM_BATCH4.md",
        """# Handoff: Uniform Batch=4

Use only `uniform_train_batch4_v1`. Begin with PRECHECK and known-OOM6 exact
preflight on the intended formal GPU. Do not run formal suites if any gate is
not PASS. Existing batch32 artifacts remain read-only and are not eligible for
batch4 readiness or aggregation.
""",
    )
    generate_scripts()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument(
        "--legacy-scope29",
        action="store_true",
        help="Explicitly generate superseded scope29 documents only.",
    )
    collect = sub.add_parser("collect-preflight")
    collect.add_argument("--suite", choices=("known_oom6", "all26"), required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if not args.legacy_scope29:
            print(
                "ERROR: legacy scope29 generation requires --legacy-scope29; "
                "the active Batch4 generator is benchmark_v2.manifest_generator.",
                file=sys.stderr,
            )
            return 74
        generate_docs()
        return 0
    report = collect_preflight(args.suite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
