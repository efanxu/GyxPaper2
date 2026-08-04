"""Explicitly isolated constants for the superseded scope29 generator.

The active Batch4 E5 contract must not import these names.  They remain
available only to the legacy document generator behind an explicit CLI flag.
"""

from __future__ import annotations


TRAINABLE_MODELS = (
    "gru",
    "dlinear",
    "lightts",
    "tide",
    "segrnn",
    "transformer",
    "patchtst",
    "itransformer",
    "timexer",
    "timesnet",
    "micn",
    "wpmixer",
    "multipatchformer",
    "timemixer",
    "tsmixer",
    "frets",
    "crossformer",
    "msgnet",
    "timefilter",
    "gcn",
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "mtgnn",
    "agcrn",
    "stid",
)
NONTRAINABLE_MODELS = ("persistence", "moving_average")
LEGACY_SCOPE_ID = "e5_scope29_legacy"


__all__ = ["LEGACY_SCOPE_ID", "NONTRAINABLE_MODELS", "TRAINABLE_MODELS"]
