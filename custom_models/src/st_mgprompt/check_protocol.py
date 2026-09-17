from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from st_mgprompt.config import DEFAULT_16_FEATURES, STMGPromptConfig, is_forbidden_input_col, resolve_project_path
from st_mgprompt.data import build_window_start_indices, compute_split_indices, make_dataloaders
from st_mgprompt.diagnostics import validate_batch_shapes, validate_model_output
from st_mgprompt.graph_prior import prepare_graph_artifacts
from st_mgprompt.metrics import evaluate_prefix_horizons
from st_mgprompt.registry import build_model


def _check(condition: bool, name: str, details: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "details": details}


def _status(condition: bool, name: str, details: str = "", warning: bool = False) -> dict[str, Any]:
    if condition:
        status = "passed"
    else:
        status = "warning" if warning else "failed"
    return {"name": name, "status": status, "passed": status != "failed", "details": details}


def run_protocol_checks(config: STMGPromptConfig, build_data: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for path_name in ["model_input_path", "eval_target_path", "location_path"]:
        path = config.resolve_path(getattr(config, path_name))
        if config.smoke and config.smoke_use_synthetic and path_name in {"model_input_path", "eval_target_path"}:
            checks.append(_check(True, f"{path_name}_exists", "smoke synthetic mode; file existence not required"))
        else:
            checks.append(_check(path.exists(), f"{path_name}_exists", str(path)))
    checks.append(_check(config.target_col == "Patv_raw", "target_is_patv_raw"))
    checks.append(_check(config.target_mask_col == "valid_target_mask", "target_mask_is_valid_target_mask"))
    checks.append(_check(config.feature_cols == DEFAULT_16_FEATURES, "default_16_features"))
    forbidden = [c for c in config.feature_cols if is_forbidden_input_col(c)]
    checks.append(_check(not forbidden, "no_forbidden_input_features", str(forbidden)))
    if config.use_msmg_dwu or config.loss_function == "msmg_dwu_loss":
        checks.append(_check(config.loss_function == "msmg_dwu_loss", "method_full_loss_name"))
        checks.append(
            _check(
                config.model_name
                in {
                    "STMGPrompt_Full_MSMGDWU",
                    "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
                    "STMGPrompt_ComponentAblation",
                },
                "msmg_dwu_model_name",
            )
        )
        checks.append(_check(config.loss_protocol == "method_full", "msmg_dwu_method_full_protocol"))
        checks.append(_check(not config.eligible_for_fair_main_table, "msmg_dwu_not_fair_main_table"))
    else:
        checks.append(_check(config.loss_function == "masked_score_aligned_hybrid", "fair_loss_name"))
        checks.append(_check(not config.use_msmg_dwu, "msmg_dwu_disabled_for_fair_protocol"))
    checks.append(_check(config.test_sample_stride == 1, "test_uses_full_windows"))
    checks.append(
        _check(
            config.decoder_input_strategy
            in {"direct_multi_output_prompt_query", "direct_multi_output_horizon_head"},
            "direct_decoder_strategy",
        )
    )
    checks.append(
        _check(
            (not config.use_stmg_coupling_block)
            or config.model_name in {
                "STMGPrompt_FairFull",
                "STMGPrompt_FairFull_Diffusion",
                "STMGPrompt_FairFull_HistoryDecoder",
                "STMGPrompt_FairFull_DiffusionHistory",
                "STMGPrompt_CouplingCrossFusion",
                "STMGPrompt_Full_MSMGDWU",
                "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
                "STMGPrompt_ComponentAblation",
            },
            "no_fake_coupling_block_for_non_coupled_models",
        )
    )
    checks.append(
        _check(
            all(c in config.feature_cols for c in config.volatility_source_cols),
            "vadsp_sources_in_base_features",
            str(config.volatility_source_cols),
        )
    )
    checks.append(_check(config.enable_physical_clip_eval, "physical_clip_eval_enabled"))
    checks.append(_check(config.physical_clip_protocol == "uniform_eval_clip_all_models", "physical_clip_uniform_protocol"))
    checks.append(_check(config.checkpoint_selection_metric == f"val_official_score_h{config.primary_val_horizon}", "checkpoint_metric_declared"))

    data_summary: dict[str, Any] | None = None
    if build_data:
        loaders = make_dataloaders(config)
        bundle = loaders["bundle"]
        batch = next(iter(bundle.train_loader))
        validate_batch_shapes(batch, config.lookback, config.max_pred_len)
        graph_data = None
        if (
            config.use_trend_prior_graph
            or config.use_graph_temporal_encoder
            or config.use_stmg_coupling_block
            or config.model_name in {
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
        ):
            graph_data = prepare_graph_artifacts(bundle, config)
        model = build_model(config, bundle.input_dim, graph_data=graph_data)
        if config.use_vadsp:
            if bundle.vadsp_robust_statistics is None:
                checks.append(_check(False, "vadsp_robust_statistics_present"))
            else:
                model.vadsp.set_robust_statistics(bundle.vadsp_robust_statistics)
                stats_shape = bundle.vadsp_robust_statistics["shape"]
                expected_shape = [bundle.num_nodes, len(config.volatility_source_cols)] if config.volatility_mode == "per_node" else [1, len(config.volatility_source_cols)]
                checks.extend(
                    [
                        _check(bool(model.vadsp.robust_stats_fitted.item()), "vadsp_robust_stats_fitted_true"),
                        _check(list(stats_shape["center"]) == expected_shape, "vadsp_stats_center_shape_correct", str(stats_shape.get("center"))),
                        _check(list(stats_shape["scale"]) == expected_shape, "vadsp_stats_scale_shape_correct", str(stats_shape.get("scale"))),
                        _check(bundle.vadsp_robust_statistics.get("fit_source") == "unique_train_timeline", "vadsp_stats_source_unique_train_timeline"),
                        _check(bundle.vadsp_robust_statistics.get("fit_split") == "train", "vadsp_stats_fit_split_train"),
                        _check(bundle.vadsp_robust_statistics.get("exact") is True, "vadsp_stats_exact_cpu"),
                        _check(bundle.vadsp_robust_statistics.get("device") == "cpu", "vadsp_stats_device_cpu"),
                        _check(bundle.vadsp_robust_statistics.get("depends_on_window_stride") is False, "vadsp_stats_stride_independent"),
                    ]
                )
        out = model(batch["x"].float())
        validate_model_output(out, batch)
        split_checks = []
        for split, (S, E) in bundle.split_indices.items():
            starts = bundle.window_indices[split]
            stride = {
                "train": config.train_sample_stride,
                "val": config.val_sample_stride,
                "test": config.test_sample_stride,
            }[split]
            expected = build_window_start_indices(S, E, config.lookback, config.max_pred_len, stride)
            contained = all(S <= t - config.lookback and t + config.max_pred_len <= E for t in starts)
            split_checks.append(
                {
                    "split": split,
                    "contained": contained,
                    "matches_builder": starts == expected,
                    "num_windows": len(starts),
                }
            )
        checks.extend(
            [
                _check(all(item["contained"] for item in split_checks), "split_contained_windows"),
                _check(all(item["matches_builder"] for item in split_checks), "prediction_start_stride_windows"),
                _check(tuple(out["pred"].shape) == tuple(batch["y"].shape), "pred_shape_bhn", str(out["pred"].shape)),
                _check(tuple(out["aux"]["pred_bnh"].shape) == (batch["x"].shape[0], batch["x"].shape[2], config.max_pred_len), "pred_convertible_bnh"),
                _check(bundle.metadata["target_mask_used_as_model_input"] is False, "mask_not_model_input"),
            ]
        )
        if config.use_vadsp:
            aux = out["aux"]
            gate = aux.get("dynamic_patch_gate")
            gate_min = float(gate.min()) if gate is not None else float(aux["dynamic_patch_gate_min"])
            gate_max = float(gate.max()) if gate is not None else float(aux["dynamic_patch_gate_max"])
            x_fine_shape = tuple(aux["x_fine"].shape) if "x_fine" in aux else tuple(aux["x_fine_shape"])
            x_coarse_shape = tuple(aux["x_coarse"].shape) if "x_coarse" in aux else tuple(aux["x_coarse_shape"])
            checks.extend(
                [
                    _check(gate is not None or "dynamic_patch_gate_mean" in aux, "vadsp_aux_has_gate_summary"),
                    _check(gate_min >= 0.0 and gate_max <= 1.0, "vadsp_gate_range_0_1"),
                    _check(x_fine_shape[:3] == tuple(batch["x"].shape[:3]), "vadsp_x_fine_shape_prefix"),
                    _check(x_coarse_shape[:3] == tuple(batch["x"].shape[:3]), "vadsp_x_coarse_shape_prefix"),
                ]
            )
            if config.vadsp_gate_mode == "fixed_dual":
                checks.extend(
                    [
                        _check(aux.get("uses_fixed_dual_granularity") is True, "fixed_dual_granularity_used"),
                        _check(abs(gate_min - 0.5) < 1e-8 and abs(gate_max - 0.5) < 1e-8, "fixed_gate_exactly_half"),
                        _check(x_fine_shape[:3] == tuple(batch["x"].shape[:3]), "fixed_x_fine_shape_prefix"),
                        _check(x_coarse_shape[:3] == tuple(batch["x"].shape[:3]), "fixed_x_coarse_shape_prefix"),
                    ]
                )
        if config.component_ablation is not None:
            checks.extend(
                [
                    _check(
                        out["aux"].get("uses_adaptive_graph") is bool(config.use_adaptive_graph),
                        "adaptive_graph_matches_variant",
                    ),
                    _check(
                        out["aux"].get("uses_trend_prior_graph") is bool(config.use_trend_prior_graph),
                        "trend_prior_matches_variant",
                    ),
                ]
            )
        if (
            config.use_trend_prior_graph
            or config.use_graph_temporal_encoder
            or config.use_stmg_coupling_block
            or config.model_name in {
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
        ):
            macro = graph_data["A_macro_trend"]
            micro = graph_data["A_micro_local"]
            checks.extend(
                [
                    _check(tuple(macro.shape) == (bundle.num_nodes, bundle.num_nodes), "macro_graph_shape_n_n"),
                    _check(tuple(micro.shape) == (bundle.num_nodes, bundle.num_nodes), "micro_graph_shape_n_n"),
                    _check(abs(float(macro.sum(axis=1).min()) - 1.0) < 1e-5, "macro_graph_row_sum_min_1"),
                    _check(abs(float(macro.sum(axis=1).max()) - 1.0) < 1e-5, "macro_graph_row_sum_max_1"),
                    _check(abs(float(micro.sum(axis=1).min()) - 1.0) < 1e-5, "micro_graph_row_sum_min_1"),
                    _check(abs(float(micro.sum(axis=1).max()) - 1.0) < 1e-5, "micro_graph_row_sum_max_1"),
                    _check(
                        bool(graph_data["metadata"]["graph_uses_train_only_statistics"])
                        is bool(config.use_trend_prior_graph),
                        "graph_train_only_statistics_matches_prior_mode",
                    ),
                    _check(not graph_data["metadata"]["graph_uses_test_statistics"], "graph_no_test_statistics"),
                    _check(
                        bool(graph_data["metadata"]["adaptive_graph_constrained_by_prior"])
                        is bool(config.use_adaptive_graph and config.use_trend_prior_graph),
                        "adaptive_graph_prior_constraint_matches_variant",
                    ),
                ]
            )
        if (
            config.use_graph_temporal_encoder
            or config.model_name in {
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
        ):
            aux = out["aux"]
            h_fine_shape = tuple(aux["h_fine"].shape) if "h_fine" in aux else tuple(aux["h_fine_shape"])
            h_coarse_shape = tuple(aux["h_coarse"].shape) if "h_coarse" in aux else tuple(aux["h_coarse_shape"])
            x_fine_shape = tuple(aux["x_fine"].shape) if "x_fine" in aux else tuple(aux["x_fine_shape"])
            x_coarse_shape = tuple(aux["x_coarse"].shape) if "x_coarse" in aux else tuple(aux["x_coarse_shape"])
            checks.extend(
                [
                    _check(aux.get("uses_graph_temporal_encoder") is True, "graph_temporal_encoder_used"),
                    _check(h_fine_shape == x_fine_shape, "h_fine_shape_matches_x_fine"),
                    _check(h_coarse_shape == x_coarse_shape, "h_coarse_shape_matches_x_coarse"),
                    _check(h_fine_shape[:3] == tuple(batch["x"].shape[:3]), "h_fine_shape_bln_prefix"),
                    _check(h_coarse_shape[:3] == tuple(batch["x"].shape[:3]), "h_coarse_shape_bln_prefix"),
                    _check(float(aux["h_fine_spatial_delta_norm"]) >= 0.0, "h_fine_delta_norm_nonnegative"),
                    _check(float(aux["h_coarse_spatial_delta_norm"]) >= 0.0, "h_coarse_delta_norm_nonnegative"),
                    _check(aux.get("uses_graph_in_temporal_encoder") == config.use_graph_in_temporal_encoder, "graph_temporal_use_graph_config"),
                    _check(aux.get("uses_temporal_attention") == config.use_temporal_attention, "temporal_attention_config"),
                ]
            )
        if config.use_stmg_coupling_block or config.model_name in {
            "STMGPrompt_FairFull",
            "STMGPrompt_FairFull_Diffusion",
            "STMGPrompt_FairFull_HistoryDecoder",
            "STMGPrompt_FairFull_DiffusionHistory",
            "STMGPrompt_CouplingCrossFusion",
            "STMGPrompt_Full_MSMGDWU",
            "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
            "STMGPrompt_ComponentAblation",
        }:
            aux = out["aux"]
            checks.extend(
                [
                    _check(aux.get("uses_stmg_coupling_block") is True, "coupling_block_used"),
                    _check(aux.get("uses_macro_prompt") is bool(config.use_macro_prompt), "macro_prompt_matches_variant"),
                    _check(aux.get("uses_cross_fusion") is bool(config.use_cross_fusion), "cross_fusion_matches_variant"),
                    _check(
                        aux.get("disable_reverse_cross") is bool(config.disable_reverse_cross),
                        "reverse_cross_direction_matches_variant",
                    ),
                    _check(
                        aux.get("disable_macro_to_fine_cross") is bool(config.disable_macro_to_fine_cross),
                        "macro_to_fine_cross_direction_matches_variant",
                    ),
                    _check(
                        aux.get("macro_to_fine_mode") == config.macro_to_fine_mode,
                        "macro_to_fine_mode_matches_variant",
                    ),
                    _check(
                        aux.get("macro_to_fine_exclude_recent_len")
                        == config.macro_to_fine_exclude_recent_len,
                        "macro_to_fine_query_coverage_matches_variant",
                    ),
                    _check(
                        aux.get("share_cross_attention_projections")
                        is bool(config.share_cross_attention_projections),
                        "cross_attention_projection_sharing_matches_variant",
                    ),
                    _check(aux.get("uses_st_prompt") is bool(config.use_st_prompt), "st_prompt_matches_variant"),
                    _check(
                        (not config.use_macro_prompt)
                        or aux.get("macro_prompt_pooling") == config.macro_prompt_pooling,
                        "macro_prompt_pooling_matches_variant",
                    ),
                    _check(
                        (not config.use_macro_prompt)
                        or aux.get("macro_prompt_attention_enabled") is (config.macro_prompt_pooling == "attention"),
                        "macro_prompt_attention_path_matches_variant",
                    ),
                    _check(
                        (not config.use_st_prompt)
                        or aux.get("st_prompt_use_node_identity") is bool(config.st_prompt_use_node_identity),
                        "st_prompt_node_identity_matches_variant",
                    ),
                    _check(
                        (not config.use_st_prompt)
                        or aux.get("st_prompt_information")
                        == (
                            "node+future+granularity"
                            if config.st_prompt_use_node_identity
                            else "future+granularity"
                        ),
                        "st_prompt_information_matches_variant",
                    ),
                    _check(
                        aux.get("macro_prompt_from_spatial_enhanced_coarse") is bool(config.use_macro_prompt),
                        "macro_prompt_source_matches_variant",
                    ),
                    _check(
                        aux.get("cross_fusion_uses_spatial_enhanced_features") is bool(config.use_cross_fusion),
                        "cross_fusion_source_matches_variant",
                    ),
                    _check(aux["coupling_metadata"].get("serial_graph_then_fusion") is False, "not_serial_graph_then_fusion"),
                    _check(aux["coupling_metadata"].get("num_coupling_layers") == config.num_coupling_layers, "num_coupling_layers_config"),
                    _check(config.fusion_mode in {"cross", "add", "concat"}, "fusion_mode_supported"),
                    _check(aux.get("uses_direct_decoder") is True, "direct_decoder_used"),
                    _check(aux.get("decoder_metadata", {}).get("teacher_forcing") is False, "no_teacher_forcing"),
                    _check(aux.get("decoder_metadata", {}).get("future_observed_features_used") is False, "decoder_no_future_observed_features"),
                    _check(
                        aux.get("decoder_metadata", {}).get("decoder_input_strategy")
                        == config.decoder_input_strategy,
                        "direct_decoder_strategy_matches_variant",
                    ),
                ]
            )
            if config.use_macro_prompt:
                checks.append(
                    _check(
                        (
                            tuple(aux["macro_prompt"].shape[:2])
                            if aux.get("macro_prompt") is not None
                            else tuple(aux["macro_prompt_shape"][:2])
                        )
                        == (batch["x"].shape[0], batch["x"].shape[2]),
                        "macro_prompt_shape_bn_prefix",
                    )
                )
            if config.use_st_prompt:
                checks.append(
                    _check(
                        (
                            tuple(aux["st_prompt"].shape[:3])
                            if aux.get("st_prompt") is not None
                            else tuple(aux["st_prompt_shape"][:3])
                        )
                        == (1, config.max_pred_len, batch["x"].shape[2]),
                        "st_prompt_shape_1hn",
                    )
                )
                if config.diagnostics_level not in {"none", "minimal"} and not config.st_prompt_use_node_identity:
                    prompt = aux.get("st_prompt")
                    checks.append(
                        _check(
                            prompt is not None
                            and torch.allclose(prompt, prompt[:, :, :1, :].expand_as(prompt)),
                            "st_prompt_shared_across_nodes_without_identity",
                        )
                    )
            if config.use_cross_fusion:
                checks.append(
                    _check(
                        0.0 <= float(aux["fusion_gate_mean"]) <= 1.0,
                        "fusion_gate_mean_range",
                    )
                )
            else:
                checks.append(_check(aux["fusion_gate_mean"] is None, "cross_fusion_strict_independent_bypass"))
            if config.graph_operator == "bidirectional_diffusion" and config.use_graph_in_temporal_encoder:
                if "micro_diffusion_forward_row_sum_min" in aux:
                    checks.extend(
                        [
                            _check(aux.get("micro_graph_operator") == "bidirectional_diffusion", "diffusion_graph_operator_used_micro"),
                            _check(aux.get("macro_graph_operator") == "bidirectional_diffusion", "diffusion_graph_operator_used_macro"),
                            _check(abs(float(aux["micro_diffusion_forward_row_sum_min"]) - 1.0) < 1e-5, "diffusion_forward_normalized_micro"),
                            _check(abs(float(aux["micro_diffusion_backward_row_sum_min"]) - 1.0) < 1e-5, "diffusion_reverse_normalized_micro"),
                            _check(int(aux["micro_diffusion_order"]) == config.diffusion_order_micro, "diffusion_order_matches_config_micro"),
                            _check(int(aux["macro_diffusion_order"]) == config.diffusion_order_macro, "diffusion_order_matches_config_macro"),
                            _check(torch.isfinite(aux["micro_beta_graph"]).all().item(), "diffusion_beta_finite_micro"),
                            _check(torch.isfinite(aux["macro_beta_graph"]).all().item(), "diffusion_beta_finite_macro"),
                            _check("micro_hop2_node_cosine_similarity" in aux, "diffusion_oversmoothing_diagnostic_present"),
                        ]
                    )
                else:
                    first_block = model.coupling_blocks[0]
                    micro_block = first_block.fine_micro_graph_temporal_encoder.block
                    macro_block = first_block.coarse_macro_graph_temporal_encoder.block
                    checks.extend(
                        [
                            _check(micro_block.graph_operator == "bidirectional_diffusion", "diffusion_module_used_micro"),
                            _check(macro_block.graph_operator == "bidirectional_diffusion", "diffusion_module_used_macro"),
                            _check(
                                int(micro_block.graph_conv.diffusion_order) == config.diffusion_order_micro,
                                "diffusion_order_matches_config_micro",
                            ),
                            _check(
                                int(macro_block.graph_conv.diffusion_order) == config.diffusion_order_macro,
                                "diffusion_order_matches_config_macro",
                            ),
                        ]
                    )
                checks.append(_check(h_fine_shape == x_fine_shape, "diffusion_output_shape_check"))
            if config.decoder_context_mode == "full_history_cross_attention":
                checks.extend(
                    [
                        _check(aux.get("decoder_metadata", {}).get("decoder_context_mode") == "full_history_cross_attention", "full_history_decoder_used"),
                        _check(int(aux["decoder_actual_history_len"]) == (config.decoder_history_len or config.lookback), "decoder_history_len_matches_config"),
                        _check(bool(aux["decoder_used_complete_history"]) is (config.decoder_history_len is None), "decoder_uses_complete_history_when_none"),
                        _check(aux.get("decoder_uses_history_only") is True, "decoder_uses_history_only"),
                        _check(aux.get("future_observed_features_used") is False, "decoder_no_future_observed_features_aux"),
                        _check(tuple(out["pred"].shape) == tuple(batch["y"].shape), "decoder_output_shape_check"),
                    ]
                )
            if config.use_msmg_dwu:
                checks.extend(
                    [
                        _check(aux["coupling_metadata"].get("loss_function") == "msmg_dwu_loss", "full_msmg_dwu_loss_metadata"),
                        _check(aux["coupling_metadata"].get("loss_changed_from_fair_protocol") is True, "full_loss_changed_from_fair_protocol"),
                        _check(aux["coupling_metadata"].get("eligible_for_fair_main_table") is False, "full_not_fair_main_metadata"),
                        _check(aux["coupling_metadata"].get("eligible_for_method_full_table") is True, "full_method_table_metadata"),
                        _check(aux["coupling_metadata"].get("fair_loss_comparison") is False, "full_not_fair_loss_comparison"),
                    ]
                )
        data_summary = {
            "input_dim": bundle.input_dim,
            "num_nodes": bundle.num_nodes,
            "split_indices": bundle.split_indices,
            "window_counts": {k: len(v) for k, v in bundle.window_indices.items()},
            "valid_target_ratio": bundle.metadata["valid_target_ratio"],
            "split_checks": split_checks,
        }

    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "checks": checks, "data_summary": data_summary}


def run_post_run_integrity(run_dir: str | Path) -> dict[str, Any]:
    run_dir = resolve_project_path(run_dir)
    checks: list[dict[str, Any]] = []
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    pred_path = run_dir / "predictions.npz"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checks.append(_status(config_path.exists(), "config_present", str(config_path)))
    checks.append(_status(metrics_path.exists(), "metrics_present", str(metrics_path)))
    checks.append(_status(pred_path.exists(), "predictions_present", str(pred_path)))
    config = STMGPromptConfig.from_json(config_path) if config_path.exists() else STMGPromptConfig()
    if pred_path.exists() and metrics_path.exists():
        arrays = np.load(pred_path)
        pred_eval = arrays["pred_kw_eval_bhn"] if "pred_kw_eval_bhn" in arrays else arrays["pred_kw_bhn"]
        pred_raw = arrays["pred_kw_raw_bhn"] if "pred_kw_raw_bhn" in arrays else pred_eval
        y = arrays["y_kw_bhn"]
        mask = arrays["mask_bhn"]
        recomputed = evaluate_prefix_horizons(
            pred_eval,
            y,
            mask,
            config.eval_horizons,
            num_nodes=config.num_nodes,
            physical_clip_applied=bool(config.enable_physical_clip_eval),
        )
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        mismatch = []
        for expected, actual in zip(recomputed, rows, strict=False):
            for key in ["MAE", "RMSE", "R2", "SMAPE", "MAPE", "official_align_score", "score", "Score"]:
                if key not in actual or actual[key] == "":
                    mismatch.append((expected["horizon"], key, "missing"))
                    continue
                if abs(float(expected[key]) - float(actual[key])) > 5e-4:
                    mismatch.append((expected["horizon"], key, float(expected[key]), float(actual[key])))
        checks.append(_status(not mismatch, "recompute_metrics_from_predictions", str(mismatch[:5])))
        score_values_ok = all(abs(float(row["official_align_score"]) - float(row["score"])) < 1e-9 and abs(float(row["score"]) - float(row["Score"])) < 1e-9 for row in rows)
        checks.append(_status(score_values_ok, "score_alias_equality_check"))
        score_unit_ok = all(float(row["Score"]) < 1000.0 for row in rows if row.get("Score"))
        checks.append(_status(score_unit_ok, "score_unit_mw_check", warning=False))
        clipped_ok = bool(np.nanmin(pred_eval) >= config.physical_power_min_kw - 1e-6 and np.nanmax(pred_eval) <= config.physical_power_max_kw + 1e-6)
        checks.append(_status(clipped_ok, "clipped_prediction_range_check"))
        raw_out_of_range = bool(np.any(pred_raw < config.physical_power_min_kw) or np.any(pred_raw > config.physical_power_max_kw))
        checks.append(_status(True, "raw_prediction_range_diagnostic", f"raw_out_of_range={raw_out_of_range}"))
    if checkpoint_path.exists():
        try:
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metric_ok = checkpoint.get("best_metric_name") == config.checkpoint_selection_metric
            checks.append(_status(metric_ok, "best_checkpoint_metric_check", str(checkpoint.get("best_metric_name"))))
            checks.append(_status(bool(checkpoint.get("checkpoint_selection", {}).get("single_checkpoint_for_all_horizons")), "single_checkpoint_for_all_horizons_check"))
        except Exception as exc:
            checks.append(_status(False, "best_checkpoint_readable", str(exc)))
    else:
        checks.append(_status(False, "best_checkpoint_present", str(checkpoint_path)))
    diag = run_dir / "diagnostics"
    graph_files = [
        "macro_prior_adjacency.npy",
        "micro_prior_adjacency.npy",
        "macro_adaptive_final.npy",
        "micro_adaptive_final.npy",
        "macro_final_adjacency.npy",
        "micro_final_adjacency.npy",
    ]
    graph_required = bool(config.use_trend_prior_graph or config.use_graph_temporal_encoder or config.use_stmg_coupling_block)
    checks.append(_status(all((diag / name).exists() for name in graph_files) or not graph_required, "final_adaptive_graph_saved_check"))
    checks.append(_status((diag / "vadsp_summary.json").exists() or not config.use_vadsp, "vadsp_gate_dynamic_range_check", warning=True))
    if (diag / "vadsp_summary.json").exists():
        vadsp = json.loads((diag / "vadsp_summary.json").read_text(encoding="utf-8"))
        checks.append(_status(float(vadsp.get("gate_high_minus_low", 0.0)) > 0.0, "vadsp_gate_physical_direction_check", warning=False))
        checks.append(_status(float(vadsp.get("gate_p99_minus_p01", 0.0)) > 1e-3, "vadsp_gate_dynamic_range_warning", warning=True))
        checks.append(_status(vadsp.get("vadsp_statistics_source") == "unique_train_timeline", "vadsp_statistics_unique_train_timeline"))
        checks.append(_status(vadsp.get("vadsp_statistics_fit_split") == "train", "vadsp_statistics_train_only"))
        checks.append(_status(vadsp.get("vadsp_statistics_exact") is True, "vadsp_statistics_exact_cpu"))
        checks.append(_status(vadsp.get("vadsp_statistics_depends_on_window_stride") is False, "vadsp_statistics_stride_independent"))
    checks.append(_status((diag / "prompt_norm.csv").exists() or not config.use_stmg_coupling_block, "prompt_diagnostic_present"))
    checks.append(_status((diag / "cross_attention_batch_summary.csv").exists() or (diag / "cross_attention_summary.csv").exists() or not config.use_stmg_coupling_block, "cross_attention_duplicate_row_check", warning=True))
    if config.graph_operator == "bidirectional_diffusion":
        checks.append(_status((diag / "diffusion_graph_summary.json").exists(), "diffusion_oversmoothing_diagnostic_present"))
        checks.append(_status((diag / "diffusion_hop_norms.csv").exists(), "diffusion_output_shape_check"))
        if (diag / "diffusion_graph_summary.json").exists():
            diffusion = json.loads((diag / "diffusion_graph_summary.json").read_text(encoding="utf-8"))
            checks.append(_status(diffusion.get("graph_operator") == "bidirectional_diffusion", "diffusion_graph_operator_used"))
            checks.append(_status(diffusion.get("diffusion_order_micro") == config.diffusion_order_micro, "diffusion_order_matches_config"))
    if config.decoder_context_mode == "full_history_cross_attention":
        checks.append(_status((diag / "decoder_attention_summary.csv").exists(), "full_history_decoder_used"))
        checks.append(_status((diag / "decoder_context_weight_summary.csv").exists(), "decoder_output_shape_check"))
        checks.append(_status(config.decoder_history_len is None, "decoder_uses_complete_history_when_none", warning=config.decoder_history_len is not None))
    checks.append(_status((run_dir / "prediction_metadata.json").exists(), "artifact_epoch_provenance_check"))
    checks.append(_status((run_dir / "model_summary.json").exists(), "efficiency_fields_present", warning=True))
    checks.append(_status((run_dir.parent / "metrics_mean_std.csv").exists() or not list(run_dir.parent.glob("seed_*")), "seed_aggregate_completeness_check", warning=True))
    failed = [item for item in checks if item["status"] == "failed"]
    warnings = [item for item in checks if item["status"] == "warning"]
    return {"passed": not failed, "num_failed": len(failed), "num_warnings": len(warnings), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ST-MGPrompt Step-0 protocol.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use-vadsp", action="store_true")
    parser.add_argument("--use-trend-prior-graph", action="store_true")
    parser.add_argument("--use-graph-temporal-encoder", action="store_true")
    parser.add_argument("--use-stmg-coupling-block", action="store_true")
    parser.add_argument("--use-msmg-dwu", action="store_true")
    parser.add_argument("--disable-graph-in-temporal-encoder", action="store_true")
    parser.add_argument("--use-temporal-attention", action="store_true")
    parser.add_argument("--no-build-data", action="store_true")
    parser.add_argument("--post-run", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output", type=str, default="custom_models/reports/protocol_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.post_run:
        if not args.run_dir:
            raise ValueError("--post-run requires --run-dir")
        report = run_post_run_integrity(args.run_dir)
        out_path = resolve_project_path(args.output) if args.output else resolve_project_path(args.run_dir) / "run_integrity_check.json"
        if args.output == "custom_models/reports/protocol_check.json":
            out_path = resolve_project_path(args.run_dir) / "run_integrity_check.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["passed"]:
            raise SystemExit(1)
        return
    config = STMGPromptConfig(smoke=args.smoke)
    if args.use_vadsp:
        config.use_vadsp = True
        config.model_name = "STMGPrompt_TemporalOnly_VADSP"
    if args.use_trend_prior_graph:
        config.use_vadsp = True
        config.use_trend_prior_graph = True
        config.use_adaptive_graph = True
        config.model_name = "STMGPrompt_TrendPriorGraph"
    if args.use_graph_temporal_encoder:
        config.use_vadsp = True
        config.use_trend_prior_graph = True
        config.use_adaptive_graph = True
        config.use_graph_temporal_encoder = True
        config.model_name = "STMGPrompt_GraphTemporalSmoke"
    if args.use_stmg_coupling_block:
        config.use_vadsp = True
        config.use_trend_prior_graph = True
        config.use_adaptive_graph = True
        config.use_graph_temporal_encoder = True
        config.use_stmg_coupling_block = True
        config.use_macro_prompt = True
        config.use_st_prompt = True
        config.model_name = "STMGPrompt_FairFull"
    if args.use_msmg_dwu:
        config.use_vadsp = True
        config.use_trend_prior_graph = True
        config.use_adaptive_graph = True
        config.use_graph_temporal_encoder = True
        config.use_stmg_coupling_block = True
        config.use_macro_prompt = True
        config.use_st_prompt = True
        config.use_msmg_dwu = True
        config.loss_function = "msmg_dwu_loss"
        config.loss_protocol = "method_full"
        config.model_name = "STMGPrompt_Full_MSMGDWU"
    if args.disable_graph_in_temporal_encoder:
        config.use_graph_in_temporal_encoder = False
    if args.use_temporal_attention:
        config.use_temporal_attention = True
    if args.smoke:
        config.lookback = 12
        config.smoke_num_time_steps = 240
        config.batch_size = 4
        config.epochs = 1
    report = run_protocol_checks(config, build_data=not args.no_build_data)
    out_path = resolve_project_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
