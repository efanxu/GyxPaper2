from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .validation import GraphProtocolError, validate_node_order


MATRIX_NAMES = (
    "A_binary_directed",
    "A_binary_undirected",
    "A_directed",
    "A_undirected",
    "A_gcn",
    "P_forward",
    "P_reverse",
    "L_sym",
    "L_tilde",
)


@dataclass(frozen=True)
class GraphSpec:
    graph_id: str
    schema_version: str
    node_count: int
    coordinate_system: str
    distance_metric: str
    elevation_used: bool
    k_selection_rule: str
    selected_k: int
    directedness: str
    self_loop_policy: Mapping[str, Any]
    weight_formula: str
    normalization_formulas: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "self_loop_policy", MappingProxyType(dict(self.self_loop_policy))
        )
        object.__setattr__(
            self,
            "normalization_formulas",
            MappingProxyType(dict(self.normalization_formulas)),
        )


@dataclass(frozen=True)
class GraphTensorBundle:
    spec: GraphSpec
    ordered_node_ids: tuple[Any, ...]
    matrices: Mapping[str, Any]
    device: str


@dataclass(frozen=True)
class GraphBundle:
    spec: GraphSpec
    ordered_node_ids: tuple[Any, ...]
    A_binary_directed: np.ndarray
    A_binary_undirected: np.ndarray
    A_directed: np.ndarray
    A_undirected: np.ndarray
    A_gcn: np.ndarray
    P_forward: np.ndarray
    P_reverse: np.ndarray
    L_sym: np.ndarray
    L_tilde: np.ndarray

    def __post_init__(self) -> None:
        for name in MATRIX_NAMES:
            matrix = np.ascontiguousarray(
                np.asarray(getattr(self, name), dtype=np.float64)
            )
            matrix.setflags(write=False)
            object.__setattr__(self, name, matrix)

    @property
    def matrices(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in MATRIX_NAMES}

    def validate_runtime_node_order(self, node_ids: Any) -> None:
        validate_node_order(
            node_ids,
            self.ordered_node_ids,
        )

    def to(self, device: Any) -> GraphTensorBundle:
        import torch

        target = torch.device(device)
        tensors = {
            name: torch.from_numpy(
                np.array(getattr(self, name), dtype=np.float64, copy=True)
            ).to(target)
            for name in MATRIX_NAMES
        }
        return GraphTensorBundle(
            spec=self.spec,
            ordered_node_ids=self.ordered_node_ids,
            matrices=MappingProxyType(tensors),
            device=str(target),
        )


@dataclass(frozen=True)
class GraphIdentity:
    graph_id: str
    node_count: int
    ordered_node_ids: tuple[Any, ...]
    selected_k: int
    graph_support_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "ordered_node_ids": list(self.ordered_node_ids),
            "selected_k": self.selected_k,
            "graph_support_names": list(self.graph_support_names),
        }


def identity_from_bundle(bundle: GraphBundle) -> GraphIdentity:
    return GraphIdentity(
        graph_id=bundle.spec.graph_id,
        node_count=bundle.spec.node_count,
        ordered_node_ids=bundle.ordered_node_ids,
        selected_k=bundle.spec.selected_k,
        graph_support_names=MATRIX_NAMES,
    )


def validate_graph_identity(
    expected: GraphIdentity | Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
) -> None:
    expected_values = (
        expected.to_dict() if isinstance(expected, GraphIdentity) else dict(expected)
    )
    required = (
        "graph_id",
        "node_count",
        "ordered_node_ids",
        "selected_k",
        "graph_support_names",
    )
    mismatches = {
        key: (actual.get(key), expected_values.get(key))
        for key in required
        if actual.get(key) != expected_values.get(key)
    }
    if mismatches:
        raise GraphProtocolError(f"{context} graph identity mismatch: {mismatches}")


def validate_native_graph_input(
    x: Any,
    *,
    runtime_node_ids: Any,
    bundle: GraphBundle,
    expected_time: int = 144,
    expected_features: int = 16,
) -> None:
    shape = tuple(int(value) for value in x.shape)
    expected_tail = (expected_time, bundle.spec.node_count, expected_features)
    if len(shape) != 4 or shape[1:] != expected_tail:
        raise GraphProtocolError(
            f"Native graph input must be (B,{expected_time},"
            f"{bundle.spec.node_count},{expected_features}), got {shape}."
        )
    bundle.validate_runtime_node_order(runtime_node_ids)
