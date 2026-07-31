from __future__ import annotations

from typing import Any

from .node_shared import NodeSharedAdapter


class NodeSharedGRUAdapter(NodeSharedAdapter):
    """Adapter for the independent one-GRU/one-head node-shared baseline."""

    def normalize_output(self, raw_output: Any, batch, **kwargs: Any):
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.semantic_trace.update(
            {
                "model_kind": "independent_node_shared_gru",
                "parameter_sharing": "one GRU and one prediction head shared by all nodes",
                "cross_node_interaction": False,
                "output_space": "normalized_Patv_raw_target_space",
            }
        )
        return output
