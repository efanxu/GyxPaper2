from __future__ import annotations

import argparse
import os
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from st_mgprompt.check_protocol import run_protocol_checks
from st_mgprompt.config import STMGPromptConfig, apply_component_ablation, resolve_project_path
from benchmark_v2.training_profiles import (
    PROFILE_ALLOWLIST,
    load_training_profile,
)
from st_mgprompt.experiment_protocol import apply_variant, assert_expected_diff, canonical_config, canonical_directory, config_diff, get_variant, write_json
from st_mgprompt.data import make_dataloaders
from st_mgprompt.diagnostics import (
    save_coupling_diagnostics,
    save_graph_snapshots,
    save_graph_temporal_diagnostics,
    model_summary,
    save_vadsp_diagnostics,
    find_cuda_tensors,
    validate_batch_shapes,
    validate_model_output,
)
from st_mgprompt.evaluate import evaluate_model
from st_mgprompt.graph_prior import prepare_graph_artifacts
from st_mgprompt.losses import get_loss_metadata
from st_mgprompt.losses import get_loss_fn
from st_mgprompt.registry import build_model, list_registered_models
from st_mgprompt.train import (
    _autocast_context,
    _make_grad_scaler,
    cuda_memory_snapshot,
    resolve_device,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-MGPrompt v4.2 Step-0 skeleton.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--canonical-full",
        action="store_true",
        help="Resolve the frozen canonical Full config without creating A0.",
    )
    parser.add_argument("--full-shape-smoke", action="store_true")
    parser.add_argument("--preflight-full-shape", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--allow-warm-start-only", action="store_true")
    parser.add_argument("--enable-faulthandler", action="store_true")
    parser.add_argument("--model-name", "--model", dest="model_name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--component-ablation",
        default=None,
        help="Formal component variant A0-A8.",
    )
    parser.add_argument(
        "--experiment-variant",
        default=None,
        help="Formal variant P0-P5/A0-A8.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-input-path", default=None)
    parser.add_argument("--eval-target-path", default=None)
    parser.add_argument("--raw-aligned-path", default=None)
    parser.add_argument("--location-path", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--target-mask-col", default=None)
    parser.add_argument("--input-patv-col", default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--max-pred-len", type=int, default=None)
    parser.add_argument("--eval-horizons", type=int, nargs="+", default=None)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument(
        "--training-profile",
        choices=PROFILE_ALLOWLIST,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early-stopping-patience", dest="patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--train-sample-stride", type=int, default=None)
    parser.add_argument("--val-sample-stride", type=int, default=None)
    parser.add_argument("--test-sample-stride", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", dest="amp_enabled", action="store_true")
    parser.add_argument("--no-amp", dest="amp_enabled", action="store_false")
    parser.set_defaults(amp_enabled=None)
    parser.add_argument("--diagnostics-level", choices=["none", "minimal", "standard", "full"], default=None)
    parser.add_argument("--prediction-accumulation", choices=["full", "streaming"], default=None)
    parser.add_argument("--windows-safe-mode", action="store_true")
    parser.add_argument("--smoke-use-real-data", action="store_true")
    parser.add_argument("--use-vadsp", action="store_true")
    parser.add_argument("--volatility-source-cols", nargs="+", default=None)
    parser.add_argument("--volatility-mode", choices=["per_node", "global"], default=None)
    parser.add_argument("--vol-window", type=int, default=None)
    parser.add_argument("--fine-kernel-size", type=int, default=None)
    parser.add_argument("--coarse-windows", type=int, nargs="+", default=None)
    parser.add_argument("--use-trend-prior-graph", action="store_true")
    parser.add_argument("--use-adaptive-graph", action="store_true")
    parser.add_argument("--graph-tag", default=None)
    parser.add_argument("--graph-output-root", default=None)
    parser.add_argument("--trend-source-col", default=None)
    parser.add_argument("--trend-method", choices=["causal_moving_average", "causal_ema", "almon_smooth"], default=None)
    parser.add_argument("--trend-window", type=int, default=None)
    parser.add_argument("--trend-ema-alpha", type=float, default=None)
    parser.add_argument("--graph-similarity-metric", choices=["pearson", "cosine"], default=None)
    parser.add_argument("--graph-top-k", type=int, default=None)
    parser.add_argument("--graph-alpha", type=float, default=None)
    parser.add_argument("--graph-operator", choices=["simple", "bidirectional_diffusion"], default=None)
    parser.add_argument("--diffusion-order-micro", type=int, default=None)
    parser.add_argument("--diffusion-order-macro", type=int, default=None)
    parser.add_argument(
        "--diffusion-direction",
        choices=["forward", "reverse", "bidirectional"],
        default=None,
    )
    parser.add_argument(
        "--diffusion-projection-mode",
        choices=["shared_output_dim"],
        default=None,
    )
    parser.add_argument("--disable-diffusion-bidirectional", action="store_true")
    parser.add_argument("--diffusion-beta-init", type=float, default=None)
    parser.add_argument("--node-embed-dim", type=int, default=None)
    parser.add_argument("--adaptive-graph-temperature", type=float, default=None)
    parser.add_argument("--use-graph-temporal-encoder", action="store_true")
    parser.add_argument("--disable-graph-in-temporal-encoder", action="store_true")
    parser.add_argument("--use-temporal-attention", action="store_true")
    parser.add_argument("--temporal-attention-heads", type=int, default=None)
    parser.add_argument("--fine-tcn-kernel-size", type=int, default=None)
    parser.add_argument("--coarse-tcn-kernel-size", type=int, default=None)
    parser.add_argument("--fine-tcn-dilations", type=int, nargs="+", default=None)
    parser.add_argument("--coarse-tcn-dilations", type=int, nargs="+", default=None)
    parser.add_argument("--use-stmg-coupling-block", action="store_true")
    parser.add_argument("--num-coupling-layers", type=int, default=None)
    parser.add_argument("--use-macro-prompt", action="store_true")
    parser.add_argument("--use-st-prompt", action="store_true")
    parser.add_argument("--decoder-context-mode", choices=["last_state", "full_history_cross_attention"], default=None)
    parser.add_argument("--decoder-history-len", type=int, default=None)
    parser.add_argument("--macro-prompt-len", type=int, default=None)
    parser.add_argument("--cross-attention-heads", type=int, default=None)
    parser.add_argument("--cross-fusion-recent-len", type=int, default=None)
    parser.add_argument("--fusion-mode", choices=["cross", "add", "concat"], default=None)
    parser.add_argument("--disable-reverse-cross", action="store_true")
    parser.add_argument("--disable-macro-to-fine-cross", action="store_true")
    parser.add_argument("--enable-physical-clip-eval", action="store_true")
    parser.add_argument("--disable-physical-clip-eval", action="store_true")
    parser.add_argument("--physical-power-min-kw", type=float, default=None)
    parser.add_argument("--physical-power-max-kw", type=float, default=None)
    parser.add_argument("--checkpoint-selection-metric", default=None)
    parser.add_argument("--primary-val-horizon", type=int, default=None)
    parser.add_argument("--macro-graph-source", default=None)
    parser.add_argument("--micro-graph-source", default=None)
    parser.add_argument("--macro-top-k", type=int, default=None)
    parser.add_argument("--micro-top-k", type=int, default=None)
    parser.add_argument("--macro-alpha", type=float, default=None)
    parser.add_argument("--micro-alpha", type=float, default=None)
    parser.add_argument("--macro-similarity-metric", choices=["pearson", "cosine"], default=None)
    parser.add_argument("--micro-similarity-metric", choices=["pearson", "cosine"], default=None)
    parser.add_argument("--graph-coordinate-cols", nargs="+", default=None)
    parser.add_argument("--graph-use-elevation", action="store_true")
    parser.add_argument("--use-msmg-dwu", action="store_true")
    parser.add_argument("--loss-protocol", choices=["fair_main", "method_full"], default=None)
    parser.add_argument("--msmg-base-loss", choices=["mae", "mse", "smooth_l1"], default=None)
    parser.add_argument("--msmg-lambda-site", type=float, default=None)
    parser.add_argument("--msmg-ema-alpha", type=float, default=None)
    parser.add_argument("--msmg-node-weight-clip", type=float, nargs=2, default=None)
    parser.add_argument("--granularity-weight-mode", choices=["uncertainty_precision", "difficulty_rate"], default=None)
    parser.add_argument("--site-weight-mode", choices=["static", "dynamic"], default=None)
    parser.add_argument(
        "--vadsp-gate-mode",
        choices=["dynamic", "fine_only", "coarse_only", "fixed_dual", "random_gate", "shuffled_volatility"],
        default=None,
    )
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> STMGPromptConfig:
    cfg = STMGPromptConfig()
    full_shape_requested = bool(getattr(args, "full_shape_smoke", False) or getattr(args, "preflight_full_shape", False))
    for key in [
        "model_name",
        "run_id",
        "model_input_path",
        "eval_target_path",
        "raw_aligned_path",
        "location_path",
        "target_col",
        "target_mask_col",
        "input_patv_col",
        "lookback",
        "max_pred_len",
        "eval_horizons",
        "split_ratios",
        "batch_size",
        "eval_batch_size",
        "train_batch_size",
        "val_batch_size",
        "test_batch_size",
        "epochs",
        "patience",
        "early_stopping_min_delta",
        "lr",
        "weight_decay",
        "hidden_dim",
        "dropout",
        "seed",
        "physical_power_min_kw",
        "physical_power_max_kw",
        "checkpoint_selection_metric",
        "primary_val_horizon",
        "train_sample_stride",
        "val_sample_stride",
        "test_sample_stride",
        "device",
        "output_root",
        "num_workers",
        "diagnostics_level",
        "prediction_accumulation",
        "volatility_source_cols",
        "volatility_mode",
        "vol_window",
        "fine_kernel_size",
        "coarse_windows",
        "graph_tag",
        "graph_output_root",
        "trend_source_col",
        "trend_method",
        "trend_window",
        "trend_ema_alpha",
        "graph_similarity_metric",
        "graph_top_k",
        "graph_alpha",
        "graph_operator",
        "diffusion_order_micro",
        "diffusion_order_macro",
        "diffusion_direction",
        "diffusion_projection_mode",
        "diffusion_beta_init",
        "node_embed_dim",
        "adaptive_graph_temperature",
        "temporal_attention_heads",
        "fine_tcn_kernel_size",
        "coarse_tcn_kernel_size",
        "fine_tcn_dilations",
        "coarse_tcn_dilations",
        "num_coupling_layers",
        "decoder_context_mode",
        "decoder_history_len",
        "macro_prompt_len",
        "cross_attention_heads",
        "cross_fusion_recent_len",
        "fusion_mode",
        "macro_graph_source",
        "micro_graph_source",
        "macro_top_k",
        "micro_top_k",
        "macro_alpha",
        "micro_alpha",
        "macro_similarity_metric",
        "micro_similarity_metric",
        "graph_coordinate_cols",
        "loss_protocol",
        "msmg_base_loss",
        "msmg_lambda_site",
        "msmg_ema_alpha",
        "msmg_node_weight_clip",
        "granularity_weight_mode",
        "site_weight_mode",
        "vadsp_gate_mode",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, tuple(value) if key == "msmg_node_weight_clip" else value)
    if args.persistent_workers:
        cfg.persistent_workers = True
    if args.pin_memory:
        cfg.pin_memory = True
    if args.amp_enabled is not None:
        cfg.amp_enabled = bool(args.amp_enabled)
        cfg.amp_dtype = "float16" if cfg.amp_enabled else cfg.amp_dtype
    if args.windows_safe_mode:
        cfg.windows_safe_mode = True
        cfg.num_workers = 0
        cfg.persistent_workers = False
        cfg.pin_memory = True
        if args.diagnostics_level is None:
            cfg.diagnostics_level = "minimal"
        if args.prediction_accumulation is None:
            cfg.prediction_accumulation = "streaming"
    if args.use_vadsp or cfg.model_name in {"STMGPrompt_TemporalOnly_VADSP", "STMGPrompt_DynamicPatching"}:
        cfg.use_vadsp = True
        if args.model_name is None:
            cfg.model_name = "STMGPrompt_TemporalOnly_VADSP"
    if args.use_trend_prior_graph or cfg.model_name == "STMGPrompt_TrendPriorGraph":
        cfg.use_vadsp = True
        cfg.use_trend_prior_graph = True
        cfg.use_adaptive_graph = True
        if args.model_name is None:
            cfg.model_name = "STMGPrompt_TrendPriorGraph"
    if args.use_adaptive_graph:
        cfg.use_adaptive_graph = True
    if args.use_graph_temporal_encoder or cfg.model_name == "STMGPrompt_GraphTemporalSmoke":
        cfg.use_vadsp = True
        cfg.use_trend_prior_graph = True
        cfg.use_adaptive_graph = True
        cfg.use_graph_temporal_encoder = True
        if args.model_name is None:
            cfg.model_name = "STMGPrompt_GraphTemporalSmoke"
    if args.disable_graph_in_temporal_encoder:
        cfg.use_graph_in_temporal_encoder = False
    if args.use_temporal_attention:
        cfg.use_temporal_attention = True
    coupled_model_names = {
        "STMGPrompt_FairFull",
        "STMGPrompt_FairFull_Diffusion",
        "STMGPrompt_FairFull_HistoryDecoder",
        "STMGPrompt_FairFull_DiffusionHistory",
        "STMGPrompt_CouplingCrossFusion",
        "STMGPrompt_Full_MSMGDWU",
        "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
    }
    if args.use_stmg_coupling_block or cfg.model_name in coupled_model_names:
        cfg.use_vadsp = True
        cfg.use_trend_prior_graph = True
        cfg.use_adaptive_graph = True
        cfg.use_graph_temporal_encoder = True
        cfg.use_stmg_coupling_block = True
        cfg.use_macro_prompt = True
        cfg.use_st_prompt = True
        if args.model_name is None:
            cfg.model_name = "STMGPrompt_FairFull"
    if cfg.model_name in {"STMGPrompt_FairFull_Diffusion", "STMGPrompt_FairFull_DiffusionHistory", "STMGPrompt_Full_MSMGDWU_DiffusionHistory"}:
        cfg.graph_operator = "bidirectional_diffusion"
    if cfg.model_name in {"STMGPrompt_FairFull_HistoryDecoder", "STMGPrompt_FairFull_DiffusionHistory", "STMGPrompt_Full_MSMGDWU_DiffusionHistory"}:
        cfg.decoder_context_mode = "full_history_cross_attention"
        if args.decoder_history_len is None:
            cfg.decoder_history_len = None
    if args.use_msmg_dwu or cfg.model_name in {"STMGPrompt_Full_MSMGDWU", "STMGPrompt_Full_MSMGDWU_DiffusionHistory"}:
        if cfg.model_name not in {"STMGPrompt_Full_MSMGDWU", "STMGPrompt_Full_MSMGDWU_DiffusionHistory"}:
            cfg.model_name = "STMGPrompt_Full_MSMGDWU"
        cfg.use_vadsp = True
        cfg.use_trend_prior_graph = True
        cfg.use_adaptive_graph = True
        cfg.use_graph_temporal_encoder = True
        cfg.use_stmg_coupling_block = True
        cfg.use_macro_prompt = True
        cfg.use_st_prompt = True
        cfg.use_msmg_dwu = True
        cfg.loss_function = "msmg_dwu_loss"
        cfg.loss_protocol = "method_full"
    if args.disable_diffusion_bidirectional:
        cfg.diffusion_use_bidirectional = False
        cfg.diffusion_direction = "forward"
    elif args.diffusion_direction is not None:
        cfg.diffusion_use_bidirectional = cfg.diffusion_direction == "bidirectional"
    if args.use_macro_prompt:
        cfg.use_macro_prompt = True
    if args.use_st_prompt:
        cfg.use_st_prompt = True
    if args.disable_reverse_cross:
        cfg.disable_reverse_cross = True
    if args.disable_macro_to_fine_cross:
        cfg.disable_macro_to_fine_cross = True
    if args.enable_physical_clip_eval:
        cfg.enable_physical_clip_eval = True
    if args.disable_physical_clip_eval:
        cfg.enable_physical_clip_eval = False
    if args.graph_use_elevation:
        cfg.graph_use_elevation = True
    formal_variant = args.experiment_variant or args.component_ablation
    if args.experiment_variant and args.component_ablation:
        raise ValueError("Use only one of --experiment-variant and --component-ablation.")
    if formal_variant is not None:
        family = "component_ablation" if str(formal_variant).upper().startswith("A") else "precision"
        cfg = apply_variant(cfg, formal_variant, family=family)
    if args.canonical_full:
        if formal_variant is not None:
            raise ValueError(
                "--canonical-full cannot be combined with an ablation variant."
            )
        cfg = canonical_config(cfg)
    if args.smoke:
        cfg.smoke = True
        cfg.smoke_use_synthetic = not args.smoke_use_real_data
        cfg.lookback = min(cfg.lookback, 12)
        cfg.batch_size = min(cfg.batch_size, 4)
        cfg.eval_batch_size = min(cfg.eval_batch_size, 4)
        cfg.hidden_dim = min(cfg.hidden_dim, 16)
        cfg.epochs = 1 if args.epochs is None else cfg.epochs
        cfg.patience = 1 if args.patience is None else cfg.patience
        cfg.train_sample_stride = 2
        cfg.val_sample_stride = 1
        cfg.test_sample_stride = 1
        cfg.smoke_num_time_steps = max(cfg.smoke_num_time_steps, 240)
    if full_shape_requested:
        cfg.smoke = False
        cfg.smoke_use_synthetic = False
        # apply_component_ablation() deliberately assigns the shared formal
        # component-ablation registry name. A full-shape preflight must keep
        # that name or config validation rejects the valid A0-A8 protocol.
        if args.model_name is None and args.component_ablation is None and args.experiment_variant is None:
            cfg.model_name = "STMGPrompt_FairFull"
        if cfg.model_name == "STMGPrompt_Full_MSMGDWU_DiffusionHistory":
            cfg.use_vadsp = True
            cfg.use_trend_prior_graph = True
            cfg.use_adaptive_graph = True
            cfg.use_graph_temporal_encoder = True
            cfg.use_stmg_coupling_block = True
            cfg.use_macro_prompt = True
            cfg.use_st_prompt = True
            cfg.use_msmg_dwu = True
            cfg.loss_function = "msmg_dwu_loss"
            cfg.loss_protocol = "method_full"
            cfg.graph_operator = "bidirectional_diffusion"
            cfg.decoder_context_mode = "full_history_cross_attention"
            cfg.decoder_history_len = None
            cfg.granularity_weight_mode = "difficulty_rate" if args.granularity_weight_mode is None else cfg.granularity_weight_mode
        cfg.lookback = 144 if args.lookback is None else cfg.lookback
        cfg.max_pred_len = 10 if args.max_pred_len is None else cfg.max_pred_len
        cfg.eval_horizons = [3, 6, 10] if args.eval_horizons is None else cfg.eval_horizons
        cfg.batch_size = 32 if args.batch_size is None else cfg.batch_size
        cfg.eval_batch_size = 64 if args.eval_batch_size is None else cfg.eval_batch_size
        if args.amp_enabled is None:
            cfg.amp_enabled = True
        if args.diagnostics_level is None:
            cfg.diagnostics_level = "minimal"
    is_a8_batch4 = str(formal_variant or "").upper() == "A8" and args.training_profile == "uniform_train_batch4_v1"
    batch_profile = load_training_profile(args.training_profile)
    if batch_profile is not None:
        requested_batch_values = {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "test_batch_size": args.test_batch_size,
        }
        invalid = {
            key: value
            for key, value in requested_batch_values.items()
            if value is not None and int(value) != 4
        }
        if invalid:
            raise ValueError(
                f"{batch_profile.profile_id} rejects conflicting batch flags: {invalid}"
            )
        if args.amp_enabled is False and not is_a8_batch4:
            raise ValueError(
                f"{batch_profile.profile_id} requires AMP=true."
            )
        cfg.batch_size = batch_profile.train_batch_size
        cfg.eval_batch_size = batch_profile.val_batch_size
        cfg.train_batch_size = batch_profile.train_batch_size
        cfg.val_batch_size = batch_profile.val_batch_size
        cfg.test_batch_size = batch_profile.test_batch_size
        cfg.amp_enabled = True
        cfg.amp_dtype = "float16"
        for key, value in batch_profile.identity().items():
            setattr(cfg, key, value)
    if is_a8_batch4:
        # The shared Batch4 profile fixes the loader batch sizes.  Formal A8
        # has its own precision identity: fp32, with AMP disabled.  This is
        # applied after the generic profile so the A8 contract cannot be
        # accidentally changed by a profile default.
        from st_mgprompt.a8_batch4_contract import (
            A8_DEFINITION,
            A8_MODEL_ID,
            A8_VARIANT,
            LOSS_ID,
            LOSS_PROTOCOL,
            PRECISION_POLICY,
            TRAINING_ROLE,
        )

        cfg.variant = A8_VARIANT
        cfg.model_id = A8_MODEL_ID
        cfg.definition = A8_DEFINITION
        cfg.component_ablation = A8_VARIANT
        cfg.model_name = "STMGPrompt_ComponentAblation"
        cfg.use_msmg_dwu = False
        cfg.loss_function = LOSS_ID
        cfg.loss_protocol = LOSS_PROTOCOL
        cfg.amp_enabled = False
        cfg.precision_policy = PRECISION_POLICY
        cfg.training_role = TRAINING_ROLE
        cfg.checkpoint_copied = False
        cfg.metrics_copied = False
        cfg.warm_started_from_historical_a8 = False
    if cfg.run_id is None:
        prefix = "full_shape_smoke" if full_shape_requested else ("smoke" if cfg.smoke else "full")
        cfg.run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return cfg


def _run_dir(cfg: STMGPromptConfig) -> Path:
    return cfg.resolve_path(cfg.output_root) / str(cfg.run_id) / cfg.model_name


def _training_batch_identity(cfg: STMGPromptConfig) -> dict:
    keys = (
        "training_batch_profile_id",
        "train_batch_size",
        "val_batch_size",
        "test_batch_size",
        "gradient_accumulation_steps",
        "effective_train_batch_size",
    )
    return {
        key: getattr(cfg, key)
        for key in keys
        if hasattr(cfg, key)
    }


def _write_training_batch_identity(run_dir: Path, cfg: STMGPromptConfig) -> None:
    identity = _training_batch_identity(cfg)
    if not identity:
        return
    (run_dir / "training_batch_profile.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    effective = {**cfg.to_dict(), **identity}
    if _is_formal_a8_batch4(cfg):
        from st_mgprompt.a8_batch4_contract import A8_SCOPE_ID

        effective.update({"scope_id": A8_SCOPE_ID, "formal_training": True})
    (run_dir / "effective_config.json").write_text(
        json.dumps(effective, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = {
        "schema_version": "st_mgprompt_uniform_batch4_artifact_v1",
        "run_id": cfg.run_id,
        "model_name": cfg.model_name,
        "component_ablation": cfg.component_ablation,
        "variant": getattr(cfg, "variant", None),
        "definition": getattr(cfg, "definition", None),
        "training_role": getattr(cfg, "training_role", None),
        "training_batch_profile_id": getattr(cfg, "training_batch_profile_id", None),
        "checkpoint_copied": bool(getattr(cfg, "checkpoint_copied", False)),
        "metrics_copied": bool(getattr(cfg, "metrics_copied", False)),
        "warm_started_from_historical_a8": bool(
            getattr(cfg, "warm_started_from_historical_a8", False)
        ),
        "precision_policy": getattr(cfg, "precision_policy", None),
        "canonical_status": (
            "BATCH4_CANONICAL_CANDIDATE"
            if cfg.component_ablation in (None, "A0")
            else "BATCH4_COMPONENT_ABLATION"
        ),
        **identity,
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_formal_a8_batch4(cfg: STMGPromptConfig) -> bool:
    """Return true only for the current, non-smoke A8 Batch4 namespace."""

    from st_mgprompt.a8_batch4_contract import (
        A8_DEFINITION,
        A8_MODEL_ID,
        A8_OUTPUT_ROOT,
        A8_RUN_ID,
        A8_VARIANT,
        TRAINING_PROFILE_ID,
    )

    try:
        configured_root = resolve_project_path(cfg.output_root).resolve()
        canonical_root = resolve_project_path(A8_OUTPUT_ROOT).resolve()
    except (OSError, TypeError, ValueError):
        return False
    return bool(
        not getattr(cfg, "smoke", False)
        and str(getattr(cfg, "run_id", "")) == A8_RUN_ID
        and str(getattr(cfg, "component_ablation", "")).upper() == A8_VARIANT
        and str(getattr(cfg, "variant", "")).upper() == A8_VARIANT
        and getattr(cfg, "model_id", None) == A8_MODEL_ID
        and getattr(cfg, "definition", None) == A8_DEFINITION
        and getattr(cfg, "training_batch_profile_id", None) == TRAINING_PROFILE_ID
        and getattr(cfg, "model_name", None) == "STMGPrompt_ComponentAblation"
        and configured_root == canonical_root
    )


def _a8_batch4_effective_config_diff(cfg: STMGPromptConfig) -> dict:
    """Prove A8 architecture identity while recording its profile overrides."""

    result = config_diff(cfg, "A8", "component_ablation")
    profile_fields = {"amp_enabled", "train_batch_size"}
    unexpected = sorted(
        set(result["unexpected_diff_fields"]) - profile_fields
    )
    missing = list(result["missing_expected_diff_fields"])
    if unexpected or missing:
        raise RuntimeError(
            "A8 Batch4 effective config differs from the formal protocol: "
            f"unexpected={unexpected}, missing={missing}"
        )
    result["expected_diff_fields"] = list(
        dict.fromkeys(
            [*result["expected_diff_fields"], *sorted(profile_fields)]
        )
    )
    result["allowed_training_profile_diff_fields"] = sorted(profile_fields)
    result["unexpected_diff_fields"] = unexpected
    result["unexpected_effective_diff_count"] = len(unexpected)
    result["passed"] = True
    return result


def _a8_relative_project_path(configured_path: str | Path) -> str:
    """Resolve a dataset path while returning only a repository-relative path."""

    project_root = Path(__file__).resolve().parents[3]
    resolved = resolve_project_path(configured_path).resolve()
    try:
        relative = resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"A8 data path escapes the repository: {configured_path}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"A8 data path is not a safe relative path: {configured_path}")
    return relative.as_posix()


def _write_a8_data_signature(
    cfg: STMGPromptConfig,
    data,
    run_dir: Path,
) -> Path:
    """Write the deterministic data identity after the real bundle is built."""

    from st_mgprompt.a8_batch4_contract import (
        A8_RUN_ID,
        DATASET_ID,
        DATA_SIGNATURE_SCHEMA_VERSION,
        DATA_SPLIT_RATIOS,
        DATA_STRIDES,
        FEATURE_ORDER,
        INPUT_PATV_COL,
        INPUT_RELATIVE_PATH,
        TARGET_COL,
        TARGET_MASK_COL,
        TARGET_RELATIVE_PATH,
        TRAINING_PROFILE_ID,
    )

    input_relative_path = _a8_relative_project_path(cfg.model_input_path)
    target_relative_path = _a8_relative_project_path(cfg.eval_target_path)
    if input_relative_path != INPUT_RELATIVE_PATH:
        raise ValueError(
            f"Formal A8 input path is not the frozen Batch4 path: {input_relative_path}"
        )
    if target_relative_path != TARGET_RELATIVE_PATH:
        raise ValueError(
            f"Formal A8 target path is not the frozen Batch4 path: {target_relative_path}"
        )
    feature_order = list(data.feature_cols)
    if feature_order != list(FEATURE_ORDER):
        raise ValueError(f"Formal A8 feature order mismatch: {feature_order}")
    if int(data.num_nodes) != 134:
        raise ValueError(f"Formal A8 node count mismatch: {data.num_nodes}")
    profile = load_training_profile(TRAINING_PROFILE_ID)
    if profile is None:
        raise ValueError(f"Missing A8 training profile: {TRAINING_PROFILE_ID}")
    signature = {
        "schema_version": DATA_SIGNATURE_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "run_id": A8_RUN_ID,
        "node_count": int(data.num_nodes),
        "feature_order": feature_order,
        "feature_names": feature_order,
        "input_relative_path": input_relative_path,
        "target_relative_path": target_relative_path,
        "input_path": input_relative_path,
        "target_path": target_relative_path,
        "target_col": cfg.target_col,
        "input_patv_col": cfg.input_patv_col,
        "target_mask_col": cfg.target_mask_col,
        "split_ratios": list(cfg.split_ratios),
        "lookback": int(cfg.lookback),
        "max_pred_len": int(cfg.max_pred_len),
        "eval_horizons": [int(value) for value in cfg.eval_horizons],
        "stride": dict(DATA_STRIDES),
        "strides": dict(DATA_STRIDES),
        "seed": int(cfg.seed),
        "training_profile_id": profile.profile_id,
    }
    if signature["split_ratios"] != list(DATA_SPLIT_RATIOS):
        raise ValueError("Formal A8 split ratios do not match the frozen protocol.")
    if signature["target_col"] != TARGET_COL:
        raise ValueError("Formal A8 target column does not match the frozen protocol.")
    if signature["input_patv_col"] != INPUT_PATV_COL:
        raise ValueError("Formal A8 input Patv column does not match the frozen protocol.")
    if signature["target_mask_col"] != TARGET_MASK_COL:
        raise ValueError("Formal A8 target mask column does not match the frozen protocol.")
    if signature["lookback"] != 144 or signature["max_pred_len"] != 10:
        raise ValueError("Formal A8 window identity does not match the frozen protocol.")
    if signature["eval_horizons"] != [3, 6, 10] or signature["seed"] != 2026:
        raise ValueError("Formal A8 horizon/seed identity does not match the frozen protocol.")
    path = run_dir / "data_signature.json"
    _atomic_json_write(path, signature, sort_keys=True)
    return path


def _write_a8_execution_receipt(
    cfg: STMGPromptConfig,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    started_at: str,
    finished_at: str,
    exit_code: int,
) -> Path:
    """Write the A8 receipt once, from the actual completed artifact files."""

    from st_mgprompt.a8_batch4_contract import (
        A8_DEFINITION,
        A8_MODEL_ID,
        A8_OUTPUT_ROOT,
        A8_RUN_ID,
        A8_SCOPE_ID,
        A8_VARIANT,
        TRAINING_ROLE,
        graph_identity,
        loss_identity,
        precision_identity,
    )

    receipt_path = run_dir / "a8_batch4_execution_receipt.json"
    if receipt_path.exists():
        raise RuntimeError(f"Refusing to overwrite an existing A8 execution receipt: {receipt_path}")
    required_files = (
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "metrics.csv",
        "metrics_eval_h3.json",
        "metrics_eval_h6.json",
        "metrics_eval_h10.json",
        "train_log.csv",
    )
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot issue A8 receipt; required files are missing: {missing}")
    dataset_signature_path = run_dir / "data_signature.json"
    if not dataset_signature_path.is_file():
        raise RuntimeError("Cannot issue A8 receipt; data_signature.json is missing.")
    dataset_signature = json.loads(dataset_signature_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_signature, dict):
        raise RuntimeError("Cannot issue A8 receipt; data_signature.json is not an object.")
    graph = graph_identity()
    loss = loss_identity()
    precision = precision_identity()
    receipt = {
        "schema_version": "st_mgprompt_a8_batch4_execution_receipt_v2",
        "status": "COMPLETED" if exit_code == 0 else "FAILED",
        "scope_id": A8_SCOPE_ID,
        "training_role": TRAINING_ROLE,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "definition": A8_DEFINITION,
        "run_id": A8_RUN_ID,
        "action": "formal_training",
        "source": str(run_dir),
        "target": str(run_dir),
        "training_profile_id": getattr(cfg, "training_batch_profile_id", "uniform_train_batch4_v1"),
        "data_signature_path": "data_signature.json",
        "batch_identity": _training_batch_identity(cfg),
        "macro_graph_identity": graph,
        "micro_graph_identity": graph,
        "loss_identity": loss,
        "precision_identity": precision,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": int(exit_code),
        "file_count": len(required_files),
        "total_size_bytes": sum((run_dir / name).stat().st_size for name in required_files),
        "message": "A8 formal artifacts completed and passed explicit checks.",
        "checkpoint_copied": False,
        "metrics_copied": False,
        "warm_started_from_historical_a8": False,
    }
    atomic_path = receipt_path.with_name(receipt_path.name + ".tmp")
    atomic_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(atomic_path, receipt_path)
    return receipt_path


def _seed_run_dir(cfg: STMGPromptConfig, seed: int) -> Path:
    return cfg.resolve_path(cfg.output_root) / str(cfg.run_id) / f"seed_{seed}" / cfg.model_name


def _graph_required(cfg: STMGPromptConfig) -> bool:
    return (
        cfg.use_trend_prior_graph
        or cfg.use_graph_temporal_encoder
        or cfg.use_stmg_coupling_block
        or cfg.model_name in {
            "STMGPrompt_TrendPriorGraph",
            "STMGPrompt_GraphTemporalSmoke",
            "STMGPrompt_FairFull",
            "STMGPrompt_FairFull_Diffusion",
            "STMGPrompt_FairFull_HistoryDecoder",
            "STMGPrompt_FairFull_DiffusionHistory",
            "STMGPrompt_CouplingCrossFusion",
            "STMGPrompt_Full_MSMGDWU",
            "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
            "STMGPrompt_ComponentAblation",
        }
    )


def _build_graph_data(cfg: STMGPromptConfig, data):
    if cfg.smoke:
        safe_run_id = str(cfg.run_id).replace("\\", "_").replace("/", "_")
        cfg.graph_output_root = str(
            cfg.resolve_path(cfg.output_root)
            / "_smoke_graph_artifacts"
            / safe_run_id
        )
    elif str(cfg.variant or "").upper().startswith("D"):
        import torch

        checkpoint_path = canonical_directory(resolve_project_path(".")) / "best_checkpoint.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict) or not {"A_macro_prior", "A_micro_prior"}.issubset(state):
            raise ValueError(f"Canonical checkpoint lacks frozen G0 graph buffers: {checkpoint_path}")
        macro = state["A_macro_prior"].detach().cpu().numpy()
        micro = state["A_micro_prior"].detach().cpu().numpy()
        expected_shape = (int(data.num_nodes), int(data.num_nodes))
        if macro.shape != expected_shape or micro.shape != expected_shape:
            raise ValueError(
                f"Frozen G0 graph shape mismatch: macro={macro.shape}, micro={micro.shape}, expected={expected_shape}"
            )
        return {
            "A_macro_trend": macro,
            "A_micro_local": micro,
            "metadata": {
                **payload.get("graph_metadata", {}),
                "graph_identity_reference": "G0/CANONICAL",
                "graph_source": "canonical_checkpoint_buffers",
                "canonical_checkpoint": str(checkpoint_path.resolve()),
                "graph_reconstructed": False,
                "transpose_topk_recomputed": False,
            },
        }
    return prepare_graph_artifacts(data, cfg) if _graph_required(cfg) else None


def _vadsp_expected_shape(cfg: STMGPromptConfig, num_nodes: int) -> list[int]:
    return [num_nodes, len(cfg.volatility_source_cols)] if cfg.volatility_mode == "per_node" else [1, len(cfg.volatility_source_cols)]


def _validate_loaded_vadsp(cfg: STMGPromptConfig, model, data) -> None:
    vadsp = getattr(model, "vadsp", None)
    if vadsp is None:
        return
    if not getattr(vadsp, "requires_robust_statistics", True):
        return
    if not bool(vadsp.robust_stats_fitted.item()):
        raise RuntimeError("Loaded checkpoint has VADSP robust_stats_fitted=false.")
    expected = _vadsp_expected_shape(cfg, data.num_nodes)
    metadata = getattr(vadsp, "robust_statistics_metadata", {}) or {}
    allow_legacy_global_broadcast = (
        metadata.get("fit_source") == "legacy_checkpoint"
        and cfg.volatility_mode == "per_node"
    )
    for name in ["robust_center", "robust_scale", "delta_abs_iqr", "delta_log_median", "delta_log_iqr"]:
        shape = list(getattr(vadsp, name).shape)
        if (
            shape != expected
            and not (cfg.volatility_mode == "global" and shape == [1, len(cfg.volatility_source_cols)])
            and not (allow_legacy_global_broadcast and shape == [1, len(cfg.volatility_source_cols)])
        ):
            raise RuntimeError(f"Loaded VADSP buffer {name} has shape {shape}, expected {expected}.")


def load_checkpoint_model(cfg: STMGPromptConfig, data, checkpoint_path: Path):
    import torch

    device = resolve_device(cfg.device)
    graph_data = _build_graph_data(cfg, data)
    model = build_model(cfg, data.input_dim, graph_data=graph_data).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    vadsp = getattr(model, "vadsp", None)
    if vadsp is not None:
        metadata = checkpoint.get("vadsp_robust_statistics_metadata")
        if metadata is not None:
            vadsp.robust_statistics_metadata = dict(metadata)
        elif not getattr(vadsp, "robust_statistics_metadata", {}).get("fit_source"):
            vadsp.robust_statistics_metadata = {
                "fit_split": "unknown",
                "fit_source": "legacy_checkpoint",
                "exact": False,
                "device": "unknown",
                "mode": cfg.volatility_mode,
                "source_cols": list(cfg.volatility_source_cols),
                "depends_on_window_stride": True,
                "eligible_for_fair_main_table": False,
            }
        _validate_loaded_vadsp(cfg, model, data)
    return model


def _environment_summary() -> dict:
    import torch

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }


def _print_environment_summary() -> None:
    summary = _environment_summary()
    print(f"Child Python: {Path(summary['sys_executable']).resolve()}", flush=True)
    print(f"sys.prefix: {summary['sys_prefix']}", flush=True)
    print(f"Python version: {summary['python_version'].splitlines()[0]}", flush=True)
    print(f"Torch version: {summary['torch_version']}", flush=True)
    print(f"CUDA available: {summary['cuda_available']}", flush=True)
    print(f"GPU name: {summary['gpu_name'] or 'NO_GPU'}", flush=True)


def _atomic_json_write(path: Path, payload: dict, *, sort_keys: bool = False) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=sort_keys,
        ),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _cuda_memory_summary() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            driver_free, driver_total = torch.cuda.mem_get_info(device)
            return {
                "memory_unit": "bytes",
                "cuda_device_index": int(device),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "driver_free": int(driver_free),
                "driver_total": int(driver_total),
                "current_allocated": int(torch.cuda.memory_allocated(device)),
                "current_reserved": int(torch.cuda.memory_reserved(device)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
    except Exception:
        pass
    return {
        "memory_unit": "bytes",
        "cuda_device_index": None,
        "cuda_device_name": None,
        "driver_free": None,
        "driver_total": None,
        "current_allocated": None,
        "current_reserved": None,
        "peak_allocated": None,
        "peak_reserved": None,
    }


def update_run_status(run_dir: Path, stage: str, extra: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_status.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    import psutil

    payload.setdefault("pid", os.getpid())
    payload.setdefault("process_start_time", float(psutil.Process(os.getpid()).create_time()))
    event = {
        "stage": stage,
        "time": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    payload.setdefault("events", []).append(event)
    payload["current_stage"] = stage
    payload["updated_at"] = event["time"]
    payload.update(extra or {})
    _atomic_json_write(path, payload)
    print(stage, flush=True)


def _resolve_checkpoint_path(args: argparse.Namespace, run_dir: Path) -> Path:
    checkpoint = args.checkpoint
    if checkpoint in {None, "", "best"}:
        return run_dir / "best_checkpoint.pt"
    if checkpoint == "last":
        return run_dir / "last_checkpoint.pt"
    path = Path(checkpoint)
    if path.is_absolute():
        return path
    run_relative = run_dir / path
    return run_relative if run_relative.exists() else resolve_project_path(path)


def _resume_checkpoint_arg(args: argparse.Namespace, run_dir: Path) -> str | Path | None:
    if args.resume_from:
        if args.resume_from == "auto":
            return "auto"
        path = Path(args.resume_from)
        if path.is_absolute():
            return path
        run_relative = run_dir / path
        return run_relative if run_relative.exists() else resolve_project_path(path)
    if args.resume:
        candidate = run_dir / "last_checkpoint.pt"
        return "auto" if candidate.exists() else None
    return None


def _failure_payload(cfg: STMGPromptConfig, run_dir: Path, stage: str, exc: BaseException) -> dict:
    train_complete = run_dir / "train_complete.json"
    eval_complete = run_dir / "evaluation_complete.json"
    train_log = run_dir / "train_log.csv"
    last_epoch = None
    if train_log.exists():
        import csv

        with train_log.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last_epoch = rows[-1].get("epoch")
    oom_context = dict(getattr(exc, "stmg_oom_context", {}) or {})
    payload = {
        "status": "FAILED",
        "exit_code": 1,
        "failure_stage": oom_context.get("stage", stage),
        "last_completed_epoch": last_epoch,
        "best_checkpoint_exists": (run_dir / "best_checkpoint.pt").exists(),
        "last_checkpoint_exists": (run_dir / "last_checkpoint.pt").exists(),
        "train_complete_exists": train_complete.exists(),
        "evaluation_complete_exists": eval_complete.exists(),
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.persistent_workers,
        "amp_enabled": bool(cfg.amp_enabled),
        "diagnostics_level": cfg.diagnostics_level,
        "train_batch_size": cfg.train_batch_size or cfg.batch_size,
        "val_batch_size": cfg.val_batch_size or cfg.eval_batch_size,
        "test_batch_size": cfg.test_batch_size or cfg.eval_batch_size,
        **_cuda_memory_summary(),
        **oom_context,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **_training_batch_identity(cfg),
    }
    if _is_formal_a8_batch4(cfg):
        from st_mgprompt.a8_batch4_contract import A8_RUN_ID, A8_SCOPE_ID

        payload.update(
            {
                "run_mode": "formal",
                "formal_training": True,
                "artifact_profile": "TRAIN",
                "scope_id": A8_SCOPE_ID,
                "run_id": A8_RUN_ID,
            }
        )
    return payload


def _finalize_a8_formal_success(
    cfg: STMGPromptConfig,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    started_at: str,
    finished_at: str,
) -> Path:
    """Publish the receipt first, then atomically publish completed status."""

    from st_mgprompt.a8_batch4_contract import A8_RUN_ID, A8_SCOPE_ID

    receipt_path = _write_a8_execution_receipt(
        cfg,
        run_dir,
        args,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=0,
    )
    update_run_status(
        run_dir,
        "PROCESS_FINISHED",
        {
            "status": "COMPLETED",
            "exit_code": 0,
            "run_mode": "formal",
            "formal_training": True,
            "artifact_profile": "TRAIN",
            "scope_id": A8_SCOPE_ID,
            "run_id": A8_RUN_ID,
            "finished_at": finished_at,
        },
    )
    return receipt_path


def _set_model_vadsp_statistics(cfg: STMGPromptConfig, model, data) -> None:
    if getattr(model, "vadsp", None) is None:
        return
    if not getattr(model.vadsp, "requires_robust_statistics", True):
        return
    if data.vadsp_robust_statistics is None:
        raise RuntimeError("VADSP robust statistics are missing for model preflight.")
    model.vadsp.set_robust_statistics(data.vadsp_robust_statistics)
    _validate_loaded_vadsp(cfg, model, data)


def _run_full_shape_smoke(args: argparse.Namespace, cfg: STMGPromptConfig) -> dict:
    import torch

    cfg.validate()
    run_dir = _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(run_dir / "config.json")
    _write_training_batch_identity(run_dir, cfg)
    set_seed(cfg.seed)
    dataloaders = make_dataloaders(cfg)
    data = dataloaders["bundle"]
    train_batch = next(iter(data.train_loader))
    validate_batch_shapes(train_batch, cfg.lookback, cfg.max_pred_len)
    effective_train_batch_size = cfg.train_batch_size or cfg.batch_size
    if tuple(train_batch["x"].shape) != (effective_train_batch_size, cfg.lookback, data.num_nodes, data.input_dim):
        raise RuntimeError(
            f"Full-shape smoke requires one complete train batch "
            f"({effective_train_batch_size},{cfg.lookback},{data.num_nodes},{data.input_dim}), got {tuple(train_batch['x'].shape)}."
        )
    graph_data = _build_graph_data(cfg, data)
    device = resolve_device(cfg.device)
    model = build_model(cfg, data.input_dim, graph_data=graph_data).to(device)
    _set_model_vadsp_statistics(cfg, model, data)
    loss_fn = get_loss_fn(cfg.loss_function, config=cfg, num_nodes=data.num_nodes)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = _make_grad_scaler(bool(cfg.amp_enabled))

    def memory_mib() -> dict[str, float | None]:
        return {
            key: (float(value / (1024**2)) if value is not None else None)
            for key, value in cuda_memory_snapshot(device).items()
        }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    x = train_batch["x"].to(device).float()
    y = train_batch["y"].to(device).float()
    mask = train_batch["valid_target_mask"].to(device).float()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    forward_start = time.perf_counter()
    with _autocast_context(device, bool(cfg.amp_enabled)):
        out = model(x)
    actual_output_shape = list(out["pred"].shape)
    finite_output = bool(torch.isfinite(out["pred"]).all().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_time = time.perf_counter() - forward_start
    train_forward_memory = memory_mib()
    validate_model_output(out, train_batch | {"y": y})
    if not finite_output:
        raise FloatingPointError("Full-shape smoke output is not finite.")
    with torch.autocast(device_type=device.type, enabled=False):
        loss = loss_fn(out["pred"].float(), y.float(), mask.float())
    if loss is None:
        raise RuntimeError("Full-shape smoke loss returned None.")
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError("Full-shape smoke loss is not finite.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    backward_start = time.perf_counter()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_time = time.perf_counter() - backward_start
    train_backward_memory = memory_mib()

    def _grad_finite(module) -> bool:
        params = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
        return bool(params) and all(torch.isfinite(p.grad).all().item() for p in params)

    gradient_checks = {
        "model_any_grad_finite": any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
        "vadsp_grad_finite": _grad_finite(model.vadsp),
        "coupling_grad_finite": _grad_finite(model.coupling_blocks),
        "decoder_grad_finite": _grad_finite(model.direct_decoder),
        "loss_backward_compatible": True,
    }
    if cfg.graph_operator == "bidirectional_diffusion":
        first_block = model.coupling_blocks[0]
        gradient_checks["micro_diffusion_grad_finite"] = _grad_finite(first_block.fine_micro_graph_temporal_encoder.block.graph_conv)
        gradient_checks["macro_diffusion_grad_finite"] = _grad_finite(first_block.coarse_macro_graph_temporal_encoder.block.graph_conv)
    if not all(gradient_checks.values()):
        raise RuntimeError(f"Full-shape smoke gradient check failed: {gradient_checks}")
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_loss_value = float(loss.detach().cpu().item())
    del out, loss, x, y, mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    validation_grad_states: list[bool] = []
    validation_memory_growth: list[float | None] = []
    validation_cuda_details: list[dict] = []
    val_loss_values: list[float] = []
    validation_batches = 0
    with torch.inference_mode():
        for batch_index, val_batch in enumerate(data.val_loader):
            if batch_index >= 5:
                break
            val_x = val_batch["x"].to(device).float()
            val_y = val_batch["y"].to(device).float()
            val_mask = val_batch["valid_target_mask"].to(device).float()
            with _autocast_context(device, bool(cfg.amp_enabled)):
                validation_grad_states.append(torch.is_grad_enabled())
                val_out = model(val_x)
            validate_model_output(val_out, val_batch | {"y": val_y})
            with torch.autocast(device_type=device.type, enabled=False):
                val_loss = loss_fn(val_out["pred"].float(), val_y.float(), val_mask.float())
            val_loss_values.append(float(val_loss.detach().cpu().item()))
            aux_details = {key: value for key, value in val_out["aux"].items() if key != "pred_bnh"}
            validation_cuda_details.extend(find_cuda_tensors(aux_details, path=f"validation[{batch_index}].aux"))
            validation_batches += 1
            del val_out, val_loss, val_x, val_y, val_mask, aux_details
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validation_memory_growth.append(memory_mib()["current_allocated"])
    if validation_batches < 3:
        raise RuntimeError(f"Full-shape smoke requires at least 3 validation batches, got {validation_batches}.")
    validation_memory = memory_mib()
    first_fusion = model.coupling_blocks[0].symmetric_cross_fusion
    attention_diagnostics = dict(first_fusion.last_attention_diagnostics)
    summary = {
        "mode": "full_shape_smoke",
        **_environment_summary(),
        "device": str(device),
        "batch_shape": {
            "train_x": list(train_batch["x"].shape),
            "train_y": list(train_batch["y"].shape),
            "train_mask": list(train_batch["valid_target_mask"].shape),
            "num_nodes": data.num_nodes,
            "input_dim": data.input_dim,
            "lookback": cfg.lookback,
            "max_pred_len": cfg.max_pred_len,
            "train_batch_size": effective_train_batch_size,
            "val_batch_size": cfg.val_batch_size or cfg.eval_batch_size,
            "test_batch_size": cfg.test_batch_size or cfg.eval_batch_size,
        },
        "input_dtype": str(train_batch["x"].dtype),
        "amp_enabled": bool(cfg.amp_enabled),
        "diagnostics_level": cfg.diagnostics_level,
        "hidden_dim": int(cfg.hidden_dim),
        "num_coupling_layers": int(cfg.num_coupling_layers),
        "graph_operator": cfg.graph_operator,
        "decoder_context_mode": cfg.decoder_context_mode,
        "granularity_weight_mode": cfg.granularity_weight_mode,
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "loss_value": train_loss_value,
        "output_shape": actual_output_shape,
        "finite_output": finite_output,
        "val_loss_value": float(sum(val_loss_values) / len(val_loss_values)),
        "loss_finite": True,
        "gradient_checks": gradient_checks,
        "train_forward_peak_mib": train_forward_memory["peak_allocated"],
        "train_backward_peak_mib": train_backward_memory["peak_allocated"],
        "validation_peak_mib": validation_memory["peak_allocated"],
        "train_forward_memory_mib": train_forward_memory,
        "train_backward_memory_mib": train_backward_memory,
        "validation_memory_mib": validation_memory,
        "validation_grad_enabled": any(validation_grad_states),
        "validation_grad_enabled_by_batch": validation_grad_states,
        "attention_shapes": {
            key: value for key, value in attention_diagnostics.items() if "shape" in key
        },
        "attention_dtypes": {
            key: value for key, value in attention_diagnostics.items() if "dtype" in key
        },
        "attention_weights_requested": bool(attention_diagnostics.get("attention_weights_requested")),
        "entropy_called": bool(attention_diagnostics.get("entropy_called")),
        "cuda_tensors_in_validation_details": validation_cuda_details,
        "memory_growth_across_validation_batches_mib": validation_memory_growth,
        "validation_batches": validation_batches,
        "forward_time_sec": float(forward_time),
        "backward_time_sec": float(backward_time),
        "vadsp_robust_statistics_metadata": dict(model.vadsp.robust_statistics_metadata),
        "status": "passed",
        **_training_batch_identity(cfg),
    }
    checkpoint_payload = {
        "schema_version": "st_mgprompt_batch4_preflight_checkpoint_v1",
        "model_state_dict": model.state_dict(),
        "model_config": cfg.to_dict(),
        "batch_identity": _training_batch_identity(cfg),
    }
    preflight_checkpoint = run_dir / "preflight_checkpoint.pt"
    torch.save(checkpoint_payload, preflight_checkpoint)
    strict_reload = build_model(
        cfg, data.input_dim, graph_data=graph_data
    ).to("cpu")
    reloaded = torch.load(
        preflight_checkpoint, map_location="cpu", weights_only=False
    )
    strict_reload.load_state_dict(reloaded["model_state_dict"], strict=True)
    summary["strict_reload_completed"] = True
    summary["artifact_validation"] = {
        "preflight_checkpoint_exists": preflight_checkpoint.is_file(),
        "training_batch_profile_exists": (
            run_dir / "training_batch_profile.json"
        ).is_file(),
    }
    (run_dir / "full_shape_smoke_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "full_shape_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return {"run_dir": str(run_dir), "full_shape_smoke_passed": True, "summary": summary}


def _run_once(args: argparse.Namespace, cfg: STMGPromptConfig, run_dir: Path) -> dict:
    cfg.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and (args.resume or args.resume_from or args.evaluate_only):
        cfg.save_json(run_dir / "requested_config.json")
    else:
        cfg.save_json(config_path)
    _write_training_batch_identity(run_dir, cfg)
    cfg.save_json(run_dir / "active_config.json")

    dataloaders = make_dataloaders(cfg)
    data = dataloaders["bundle"]
    if _is_formal_a8_batch4(cfg):
        _write_a8_data_signature(cfg, data, run_dir)
    first_batch = next(iter(data.train_loader))
    validate_batch_shapes(first_batch, cfg.lookback, cfg.max_pred_len)

    if args.evaluate_only:
        checkpoint_path = _resolve_checkpoint_path(args, run_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        update_run_status(run_dir, "TEST_STARTED", {"checkpoint": str(checkpoint_path)})
        model = load_checkpoint_model(cfg, data, checkpoint_path)
        skipped = 0
        train_logs = []
    else:
        update_run_status(run_dir, "TRAIN_STARTED")
        model, train_logs, checkpoint_path, skipped = train_model(
            cfg,
            data,
            run_dir,
            resume_from=_resume_checkpoint_arg(args, run_dir),
            allow_warm_start_only=bool(args.allow_warm_start_only),
        )
        update_run_status(
            run_dir,
            "TRAIN_FINISHED",
            {
                "checkpoint": str(checkpoint_path),
                "last_completed_epoch": int(train_logs[-1]["epoch"]) if train_logs else None,
            },
        )
        update_run_status(run_dir, "BEST_CHECKPOINT_SAVED", {"checkpoint": str(checkpoint_path)})

    device = resolve_device(cfg.device)
    update_run_status(run_dir, "FINAL_VALIDATION_STARTED")
    validation_batch = {
        **first_batch,
        "x": first_batch["x"].to(device).float(),
        "y": first_batch["y"].to(device).float(),
    }
    import torch

    with torch.inference_mode():
        out = model(validation_batch["x"])
    validate_model_output(out, validation_batch)
    update_run_status(run_dir, "FINAL_VALIDATION_FINISHED")
    summary = model_summary(model, data)
    checkpoint_info = {}
    try:
        import torch

        checkpoint_info = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        checkpoint_info = {}
    minimal_diagnostics = cfg.diagnostics_level in {"none", "minimal"}
    graph_summary = {}
    empirical_graph_variant = str(getattr(cfg, "variant", "")).upper().startswith("G")
    if not minimal_diagnostics or empirical_graph_variant:
        update_run_status(run_dir, "DIAGNOSTICS_STARTED")
        graph_summary = save_graph_snapshots(model, run_dir / "diagnostics", checkpoint_name=Path(checkpoint_path).name)
        if minimal_diagnostics:
            update_run_status(run_dir, "DIAGNOSTICS_FINISHED")
    summary.update(
        {
            "model_name": cfg.model_name,
            "run_id": cfg.run_id,
            "seed": cfg.seed,
            "checkpoint": str(checkpoint_path),
            "loss_function": cfg.loss_function,
            "macro_prompt_len": cfg.macro_prompt_len,
            "macro_prompt_pooling": cfg.macro_prompt_pooling,
            "macro_prompt_attention_enabled": cfg.macro_prompt_pooling == "attention",
            "st_prompt_mode": cfg.st_prompt_mode,
            "st_prompt_use_node_identity": bool(cfg.st_prompt_use_node_identity),
            "st_prompt_information": (
                "node+future+granularity"
                if cfg.st_prompt_use_node_identity
                else "future+granularity"
            ),
            "decoder_input_strategy": cfg.decoder_input_strategy,
            "eligible_for_fair_main_table": cfg.eligible_for_fair_main_table,
            **get_loss_metadata(cfg),
            "skipped_all_invalid_batches": skipped,
            "registry": list_registered_models(),
            "metadata": data.metadata,
            "graph_snapshot_summary": graph_summary,
            "train_time_sec": sum(float(row.get("epoch_time_sec", 0.0)) for row in train_logs),
            "mean_epoch_time_sec": (
                sum(float(row.get("epoch_time_sec", 0.0)) for row in train_logs) / len(train_logs)
                if train_logs
                else None
            ),
            "best_epoch": checkpoint_info.get("best_epoch") if checkpoint_info else None,
            "best_metric_name": checkpoint_info.get("best_metric_name") if checkpoint_info else None,
            "best_metric_value": checkpoint_info.get("best_metric_value") if checkpoint_info else None,
            "best_val_Score_H3": checkpoint_info.get("best_val_score_h3") if checkpoint_info else None,
            "best_val_Score_H6": checkpoint_info.get("best_val_score_h6") if checkpoint_info else None,
            "best_val_Score_H10": checkpoint_info.get("best_val_score_h10") if checkpoint_info else None,
            "branch_diagnostics": {
                key: out["aux"].get(key)
                for key in [
                    "fine_branch_active",
                    "coarse_branch_active",
                    "fine_branch_active_ratio",
                    "coarse_branch_active_ratio",
                    "fine_representation_norm",
                    "coarse_representation_norm",
                    "x_fine_shape",
                    "x_coarse_shape",
                    "vadsp_mode",
                    "coarse_alignment_mode",
                    "coarse_upsample_rule",
                    "branch_graph_assignment",
                    "fine_graph_id",
                    "coarse_graph_id",
                    "branch_graph_assignment_trace",
                    "graph_prior_component",
                    "adaptive_support_mode",
                    "graph_rewire_mode",
                ]
                if key in out["aux"]
            },
        }
    )
    if not minimal_diagnostics:
        vadsp_summary = save_vadsp_diagnostics(model, data.test_loader, run_dir / "diagnostics")
        if vadsp_summary is not None:
            summary["vadsp_summary"] = vadsp_summary
        graph_temporal_summary = save_graph_temporal_diagnostics(model, data.test_loader, run_dir / "diagnostics")
        if graph_temporal_summary is not None:
            summary["graph_temporal_summary"] = graph_temporal_summary
        coupling_summary = save_coupling_diagnostics(model, data.test_loader, run_dir / "diagnostics")
        if coupling_summary is not None:
            summary["coupling_summary"] = coupling_summary
            summary["coupling_metadata"] = coupling_summary.get("metadata", {})
        update_run_status(run_dir, "DIAGNOSTICS_FINISHED")
    metrics = []
    if args.train_only or args.skip_test:
        update_run_status(run_dir, "PROCESS_FINISHED", {"skip_test": True})
    else:
        if not args.evaluate_only:
            update_run_status(run_dir, "TEST_STARTED", {"checkpoint": str(checkpoint_path)})
        metrics = evaluate_model(
            cfg,
            data,
            model,
            run_dir,
            stage_callback=lambda stage: update_run_status(
                run_dir,
                stage,
                {"checkpoint": str(checkpoint_path)},
            ),
        )
        update_run_status(run_dir, "TEST_FINISHED")
    prediction_metadata_path = run_dir / "prediction_metadata.json"
    if prediction_metadata_path.exists():
        prediction_metadata = json.loads(prediction_metadata_path.read_text(encoding="utf-8"))
        summary["inference_efficiency"] = {
            key: prediction_metadata.get(key)
            for key in [
                "inference_time_sec_full_test",
                "inference_time_ms_per_window",
                "inference_windows_per_sec",
                "inference_valid_points_per_sec",
            ]
        }
    (run_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    update_run_status(run_dir, "FINAL_ARTIFACT_VALIDATION_STARTED")
    artifact_names = [
        "best_checkpoint.pt",
        "metrics.csv",
        "metrics_eval_h3.json",
        "metrics_eval_h6.json",
        "metrics_eval_h10.json",
        "prediction_metadata.json",
        "evaluation_complete.json",
    ]
    artifact_validation = {name: (run_dir / name).exists() for name in artifact_names}
    artifacts_complete = all(artifact_validation.values())
    update_run_status(
        run_dir,
        "FINAL_ARTIFACT_VALIDATION_FINISHED",
        {"artifact_validation": artifact_validation, "artifacts_complete": artifacts_complete},
    )
    protocol_passed = True
    if not (args.train_only or args.skip_test):
        update_run_status(run_dir, "PROTOCOL_CHECK_STARTED")
        # The run already built the real data/model and completed inference. Rebuilding
        # another full graph/model here is unsafe on Windows and was the source of the
        # historical post-TEST_FINISHED native exit. Post-run checks are intentionally
        # artifact/config checks; full model preflight remains a separate smoke command.
        protocol_report = run_protocol_checks(cfg, build_data=False)
        (run_dir / "protocol_check.json").write_text(
            json.dumps(protocol_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "protocol_report.json").write_text(
            json.dumps(protocol_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        protocol_passed = bool(protocol_report["passed"])
        update_run_status(run_dir, "PROTOCOL_CHECK_FINISHED", {"protocol_passed": protocol_passed})
    update_run_status(run_dir, "PROCESS_FINISHED")
    return {"run_dir": str(run_dir), "metrics": metrics, "protocol_passed": protocol_passed}


def main() -> None:
    args = parse_args()
    if args.enable_faulthandler:
        import faulthandler

        faulthandler.enable(all_threads=True)
    if args.train_only and args.evaluate_only:
        raise ValueError("Choose only one of --train-only or --evaluate-only.")
    full_shape_requested = bool(args.full_shape_smoke or args.preflight_full_shape)
    selected_modes = [bool(args.smoke), bool(args.full), full_shape_requested]
    if sum(selected_modes) > 1:
        raise ValueError("Choose only one of --smoke, --full, or --full-shape-smoke/--preflight-full-shape.")
    if args.list_models:
        print(json.dumps(list_registered_models(), indent=2, ensure_ascii=False))
        return
    cfg = build_config(args)
    _print_environment_summary()
    formal_variant = args.experiment_variant or args.component_ablation
    is_formal_a8_batch4 = _is_formal_a8_batch4(cfg)
    if is_formal_a8_batch4 and (
        args.resume
        or args.resume_from
        or args.allow_warm_start_only
        or args.checkpoint
    ):
        raise ValueError(
            "Formal A8 Batch4 cannot resume from or load any historical checkpoint."
        )
    if formal_variant is not None:
        variant = get_variant(
            formal_variant,
            family="component_ablation" if str(formal_variant).upper().startswith("A") else "precision",
        )
        if not variant.trainable:
            raise ValueError(
                f"{variant.variant_id} is REFERENCE_ONLY and cannot be trained or evaluated as an independent run."
            )
        diff_config = cfg
        if args.smoke or full_shape_requested:
            diff_config = apply_variant(STMGPromptConfig(), variant.variant_id, variant.experiment_family)
        if is_formal_a8_batch4:
            diff = _a8_batch4_effective_config_diff(diff_config)
        else:
            diff = assert_expected_diff(diff_config, variant.variant_id, variant.experiment_family)
        run_dir = _run_dir(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "effective_config_diff.json", diff)
    if full_shape_requested:
        try:
            result = _run_full_shape_smoke(args, cfg)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as exc:
            run_dir = _run_dir(cfg)
            run_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "mode": "full_shape_smoke",
                **_environment_summary(),
                "run_dir": str(run_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (run_dir / "full_shape_smoke_failure.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
            raise
        return
    if args.seeds:
        root = cfg.resolve_path(cfg.output_root) / str(cfg.run_id)
        results = []
        for seed in args.seeds:
            seed_data = cfg.to_dict()
            seed_data.pop("eligible_for_fair_main_table", None)
            seed_cfg = STMGPromptConfig(**seed_data)
            seed_cfg.seed = int(seed)
            seed_cfg.run_id = cfg.run_id
            results.append(_run_once(args, seed_cfg, _seed_run_dir(seed_cfg, int(seed))))
        from st_mgprompt.summarize_results import write_seed_aggregate

        write_seed_aggregate(root, cfg.model_name)
        print(json.dumps({"run_root": str(root), "seed_results": results}, indent=2, ensure_ascii=False))
        if not all(item["protocol_passed"] for item in results):
            raise SystemExit(1)
    else:
        run_dir = _run_dir(cfg)
        started_at = datetime.utcnow().isoformat() + "Z"
        try:
            result = _run_once(args, cfg, run_dir)
            if (
                is_formal_a8_batch4
                and not (args.train_only or args.skip_test)
                and result["protocol_passed"]
            ):
                _finalize_a8_formal_success(
                    cfg,
                    run_dir,
                    args,
                    started_at=started_at,
                    finished_at=datetime.utcnow().isoformat() + "Z",
                )
        except Exception as exc:
            failure = _failure_payload(cfg, run_dir, "PROCESS_EXCEPTION", exc)
            update_run_status(run_dir, "PROCESS_EXCEPTION", failure)
            (run_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["protocol_passed"]:
            if is_formal_a8_batch4:
                failure = _failure_payload(
                    cfg,
                    run_dir,
                    "PROTOCOL_CHECK_FAILED",
                    RuntimeError("A8 formal protocol check did not pass."),
                )
                update_run_status(run_dir, "PROCESS_EXCEPTION", failure)
                (run_dir / "failure.json").write_text(
                    json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
