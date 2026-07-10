# draft/
# Backbone, kv_inject, moe_reused_router (Variants A/B + dense/scratch controls),
# markov_head, conf_head, and the assembled DraftModel.

from .config import DraftConfig
from .model import DraftModel

__all__ = ["DraftConfig", "DraftModel"]
