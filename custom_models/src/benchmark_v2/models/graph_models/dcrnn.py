from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ...graph import GraphBundle
from .common import graph_identity_dict, runtime_support


class DiffusionLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        support_count: int = 2,
        max_diffusion_step: int = 2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.support_count = int(support_count)
        self.max_diffusion_step = int(max_diffusion_step)
        self.basis_count = (
            1 + self.support_count * self.max_diffusion_step
        )
        self.weight = nn.Parameter(
            torch.empty(self.basis_count * self.input_dim, self.output_dim)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.last_basis: torch.Tensor | None = None

    def diffusion_basis(
        self, x: torch.Tensor, supports: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if len(supports) != self.support_count:
            raise ValueError(
                f"Expected {self.support_count} diffusion supports."
            )
        terms = [x]
        for support in supports:
            value = x
            for _ in range(self.max_diffusion_step):
                value = torch.einsum("nm,bmf->bnf", support, value)
                terms.append(value)
        return torch.stack(terms, dim=2)

    def forward(
        self, x: torch.Tensor, supports: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        basis = self.diffusion_basis(x, supports)
        self.last_basis = basis
        flattened = basis.reshape(
            x.shape[0], x.shape[1], self.basis_count * self.input_dim
        )
        return flattened @ self.weight + self.bias


class DCGRUCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        max_diffusion_step: int = 2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        joined_dim = self.input_dim + self.hidden_dim
        kwargs = {
            "support_count": 2,
            "max_diffusion_step": int(max_diffusion_step),
        }
        self.reset_diffusion = DiffusionLinear(
            joined_dim, self.hidden_dim, **kwargs
        )
        self.update_diffusion = DiffusionLinear(
            joined_dim, self.hidden_dim, **kwargs
        )
        self.candidate_diffusion = DiffusionLinear(
            joined_dim, self.hidden_dim, **kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        supports: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        joined = torch.cat((x, hidden), dim=-1)
        reset = torch.sigmoid(self.reset_diffusion(joined, supports))
        update = torch.sigmoid(self.update_diffusion(joined, supports))
        candidate_input = torch.cat((x, reset * hidden), dim=-1)
        candidate = torch.tanh(
            self.candidate_diffusion(candidate_input, supports)
        )
        return update * hidden + (1.0 - update) * candidate


class DCRNNForecast(nn.Module):
    """Dual-random-walk DCGRU encoder-decoder with a zero-GO decoder."""

    def __init__(
        self,
        *,
        bundle: GraphBundle,
        encoder_input_dim: int = 16,
        decoder_input_dim: int = 1,
        hidden_dim: int = 64,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        max_diffusion_step: int = 2,
        lookback: int = 144,
        horizon: int = 10,
    ):
        super().__init__()
        if int(num_encoder_layers) != 2 or int(num_decoder_layers) != 2:
            raise ValueError("E3-B freezes DCRNN to two encoder/decoder layers.")
        self.encoder_input_dim = int(encoder_input_dim)
        self.decoder_input_dim = int(decoder_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.max_diffusion_step = int(max_diffusion_step)
        self.encoder_cells = nn.ModuleList(
            [
                DCGRUCell(
                    self.encoder_input_dim if index == 0 else self.hidden_dim,
                    self.hidden_dim,
                    max_diffusion_step=self.max_diffusion_step,
                )
                for index in range(int(num_encoder_layers))
            ]
        )
        self.decoder_cells = nn.ModuleList(
            [
                DCGRUCell(
                    self.decoder_input_dim if index == 0 else self.hidden_dim,
                    self.hidden_dim,
                    max_diffusion_step=self.max_diffusion_step,
                )
                for index in range(int(num_decoder_layers))
            ]
        )
        self.output_projection = nn.Linear(self.hidden_dim, 1)
        self.register_buffer(
            "P_forward", runtime_support(bundle, "P_forward"), persistent=False
        )
        self.register_buffer(
            "P_reverse", runtime_support(bundle, "P_reverse"), persistent=False
        )
        self.graph_identity = graph_identity_dict(
            bundle, ("P_forward", "P_reverse")
        )
        self.teacher_forcing = False
        self.scheduled_sampling = False
        self.curriculum_learning = False
        self.last_shape_trace: dict[str, Any] = {}

    @property
    def supports(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.P_forward, self.P_reverse

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch, _, nodes, _ = x.shape
        states = [
            x.new_zeros(batch, nodes, self.hidden_dim)
            for _ in self.encoder_cells
        ]
        for time_index in range(self.lookback):
            layer_input = x[:, time_index]
            for layer_index, cell in enumerate(self.encoder_cells):
                states[layer_index] = cell(
                    layer_input, states[layer_index], self.supports
                )
                layer_input = states[layer_index]
        return states

    def decode(self, states: list[torch.Tensor]) -> torch.Tensor:
        batch, nodes, _ = states[0].shape
        decoder_input = states[0].new_zeros(
            batch, nodes, self.decoder_input_dim
        )
        decoder_states = list(states)
        predictions = []
        for _ in range(self.horizon):
            layer_input = decoder_input
            for layer_index, cell in enumerate(self.decoder_cells):
                decoder_states[layer_index] = cell(
                    layer_input,
                    decoder_states[layer_index],
                    self.supports,
                )
                layer_input = decoder_states[layer_index]
            prediction = self.output_projection(layer_input)
            predictions.append(prediction)
            decoder_input = prediction
        return torch.cat(predictions, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (
            self.lookback,
            self.P_forward.shape[0],
            self.encoder_input_dim,
        ):
            raise ValueError(
                "DCRNNForecast input mismatch: "
                f"expected (B,{self.lookback},{self.P_forward.shape[0]},"
                f"{self.encoder_input_dim}), got {tuple(x.shape)}."
            )
        encoder_states = self.encode(x)
        prediction = self.decode(encoder_states)
        self.last_shape_trace = {
            "input": list(x.shape),
            "encoder_state": [
                len(encoder_states),
                int(x.shape[0]),
                int(x.shape[2]),
                self.hidden_dim,
            ],
            "go_token": [int(x.shape[0]), int(x.shape[2]), 1],
            "decoder_steps": self.horizon,
            "prediction": list(prediction.shape),
        }
        return prediction


def create_model(
    config: Mapping[str, Any] | None = None,
    protocol: Any = None,
    *,
    graph_bundle: GraphBundle,
) -> DCRNNForecast:
    values = dict(config or {})
    return DCRNNForecast(
        bundle=graph_bundle,
        encoder_input_dim=int(values.get("encoder_input_dim", 16)),
        decoder_input_dim=int(values.get("decoder_input_dim", 1)),
        hidden_dim=int(values.get("hidden_dim", 64)),
        num_encoder_layers=int(values.get("num_encoder_layers", 2)),
        num_decoder_layers=int(values.get("num_decoder_layers", 2)),
        max_diffusion_step=int(values.get("max_diffusion_step", 2)),
        lookback=int(values.get("lookback", 144)),
        horizon=int(values.get("horizon", 10)),
    )
