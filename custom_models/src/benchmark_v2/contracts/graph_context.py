from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ContractError


@dataclass(frozen=True)
class GraphContext:
    node_order: tuple[Any, ...]
    adjacency_metadata: dict[str, Any]
    graph_hash: str
    directed: bool
    self_loop: bool
    normalization_metadata: dict[str, Any]

    def validate(self, node_count: int) -> None:
        if len(self.node_order) != node_count:
            raise ContractError(f"GraphContext node order has {len(self.node_order)} nodes, expected {node_count}")
        if not self.graph_hash or not self.adjacency_metadata:
            raise ContractError("GraphContext requires explicit adjacency metadata and graph_hash")

