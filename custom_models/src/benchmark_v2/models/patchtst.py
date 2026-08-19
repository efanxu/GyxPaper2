from __future__ import annotations

from types import MethodType
from types import SimpleNamespace
from typing import Any, Mapping

from ..upstream import load_tslib_model_class


def _checkpointed_encoder_forward(
    encoder, x, attn_mask=None, tau=None, delta=None
):
    import torch
    from torch.utils.checkpoint import checkpoint

    if encoder.conv_layers is not None:
        return encoder._benchmark_v2_original_forward(
            x, attn_mask=attn_mask, tau=tau, delta=delta
        )
    attns = []
    for attention_layer in encoder.attn_layers:
        def run_layer(value, selected_layer=attention_layer):
            return selected_layer(
                value,
                attn_mask=attn_mask,
                tau=tau,
                delta=delta,
            )[0]

        if encoder.training and torch.is_grad_enabled():
            x = checkpoint(run_layer, x, use_reentrant=False)
        else:
            x = run_layer(x)
        attns.append(None)
    if encoder.norm is not None:
        x = encoder.norm(x)
    return x, attns


def enable_e5_gradient_checkpointing(model) -> None:
    """Reduce PatchTST FP32 activation memory without changing parameters."""

    encoder = getattr(model, "encoder", None)
    if encoder is None or getattr(
        encoder, "_benchmark_v2_e5_gradient_checkpointing", False
    ):
        return
    encoder._benchmark_v2_original_forward = encoder.forward
    encoder.forward = MethodType(_checkpointed_encoder_forward, encoder)
    encoder._benchmark_v2_e5_gradient_checkpointing = True


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    values = dict(config or {})
    model_class, source = load_tslib_model_class("patchtst")
    model = model_class(
        SimpleNamespace(**values),
        patch_len=int(values["patch_len"]),
        stride=int(values["stride"]),
    )
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
