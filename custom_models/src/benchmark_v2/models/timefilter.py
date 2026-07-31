from __future__ import annotations

from types import SimpleNamespace
from types import MethodType
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..upstream import load_tslib_model_class


def _straight_through_top_p(self, x, is_training, noise_epsilon=1e-2):
    clean_logits = self.gate(x)
    if self.noisy_gating and is_training:
        raw_noise = self.noise(x)
        noise_stddev = self.softplus(raw_noise) + noise_epsilon
        routing_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
    else:
        routing_logits = clean_logits
    probabilities = self.softmax(routing_logits)
    loss_dynamic = self.cross_entropy(probabilities)
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > self.top_p
    threshold_indices = mask.long().argmax(dim=-1)
    threshold_mask = F.one_hot(
        threshold_indices, num_classes=sorted_indices.size(-1)
    ).bool()
    mask = mask & ~threshold_mask
    hard = torch.zeros_like(mask)
    zero_indices = (mask == 0).nonzero(as_tuple=True)
    hard[
        zero_indices[0],
        zero_indices[1],
        sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]],
    ] = 1
    retained = torch.where(mask, 0.0, sorted_probs)
    loss = self.cv_squared(retained.sum(0)) + 0.1 * loss_dynamic
    # The forward mask is bit-identical to upstream hard top-p routing.  The
    # straight-through term keeps the frozen masked-MSE objective able to train
    # the gate without adding the upstream MoE auxiliary loss.
    return hard + probabilities - probabilities.detach(), loss


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    model_class, source = load_tslib_model_class("timefilter")
    model = model_class(SimpleNamespace(**dict(config or {})))
    for module in model.modules():
        if (
            hasattr(module, "gate")
            and hasattr(module, "noise")
            and hasattr(module, "noisy_top_k_gating")
        ):
            module.noisy_top_k_gating = MethodType(
                _straight_through_top_p, module
            )
    model._benchmark_v2_gate_policy = (
        "upstream-identical hard top-p forward with straight-through gate "
        "gradient; MoE auxiliary loss excluded from masked_mse"
    )
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
