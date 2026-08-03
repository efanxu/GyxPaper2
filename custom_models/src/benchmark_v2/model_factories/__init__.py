"""Fixed, auditable model factory boundary.

The registry in this package contains only a closed mapping from canonical
model ids to repository-owned factory modules.  It deliberately does not
scan packages or accept import paths from callers.
"""

from .registry import get_model_factory

__all__ = ["get_model_factory"]
